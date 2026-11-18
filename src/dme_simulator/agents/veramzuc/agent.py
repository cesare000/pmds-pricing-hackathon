"""Adaptive anti-deadstock & competition-aware agent for unknown DME scenarios."""
from collections import Counter
from math import ceil, floor, isfinite

from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing


class UniversalDominatorAgent(BaseAgent):
    def __init__(self):
        self.identity: str | None = None
        self.candidate_ids: set[str] | None = None
        self.auction_cursor = 0
        self.sales_cursor = 0

        # Memoria aste
        self.bid_ratios: dict[str, float] = {}
        self.last_wanted: Counter = Counter()
        self.avg_paid_cost: dict[str, float] = {}

        # Memoria vendite e scoperta curva di domanda
        self.anchor_prices: dict[str, float] = {}
        self.verified_sales: Counter = Counter()
        self.ema_demand: dict[str, float] = {}
        self.dry_turns: Counter = Counter()
        self.initialized = False

    def _sync_history(self, state):
        if self.identity is None:
            matches = {
                team
                for team, cash in state.public_capital.items()
                if abs(cash - state.capital) < 1e-6
                and state.public_inventories.get(team) == state.inventory
            }
            if self.candidate_ids is None:
                self.candidate_ids = matches
            elif matches:
                narrowed = self.candidate_ids & matches
                if narrowed:
                    self.candidate_ids = narrowed
            if self.candidate_ids and len(self.candidate_ids) == 1:
                self.identity = next(iter(self.candidate_ids))

        wins = Counter()
        sales = Counter()
        my_max_sold: dict[str, float] = {}
        my_min_sold: dict[str, float] = {}
        my_rev: dict[str, float] = {}
        market_max: dict[str, float] = {}

        for tx in state.auction_transactions[self.auction_cursor:]:
            if self.identity is not None and tx.buyer == self.identity:
                wins[tx.good] += tx.quantity
                prev = self.avg_paid_cost.get(tx.good, tx.price)
                self.avg_paid_cost[tx.good] = 0.45 * tx.price + 0.55 * prev

        for tx in state.sales_transactions[self.sales_cursor:]:
            if tx.price > market_max.get(tx.good, 0.0):
                market_max[tx.good] = tx.price
            if self.identity is not None and tx.seller == self.identity:
                sales[tx.good] += tx.quantity
                self.verified_sales[tx.good] += tx.quantity
                my_rev[tx.good] = my_rev.get(tx.good, 0.0) + tx.quantity * tx.price
                if tx.price > my_max_sold.get(tx.good, 0.0):
                    my_max_sold[tx.good] = tx.price
                if tx.good not in my_min_sold or tx.price < my_min_sold[tx.good]:
                    my_min_sold[tx.good] = tx.price

        self.auction_cursor = len(state.auction_transactions)
        self.sales_cursor = len(state.sales_transactions)
        return wins, sales, my_max_sold, my_min_sold, my_rev, market_max

    def decide(self, state, goods_config):
        wins, sales, my_max_sold, my_min_sold, my_rev, market_max = self._sync_history(state)
        lots = {lot.good: lot for lot in state.auction_lots}
        firms = max(1, len(state.public_capital))

        # 1. Calibrazione iniziale sulla pressione competitiva reale
        if not self.initialized:
            self.initialized = True
            for mat in goods_config.raw_materials:
                lot_qty = lots[mat].quantity if mat in lots else 0
                per_firm = lot_qty / firms
                # Se ci sono pochi pezzi per squadra, parti subito competitivo contro gli umani
                if per_firm < 4.0:
                    self.bid_ratios[mat] = 1.45
                elif per_firm < 10.0:
                    self.bid_ratios[mat] = 1.25
                else:
                    self.bid_ratios[mat] = 1.10

        # 2. Indice di scarsità nel turno corrente
        scarcity: dict[str, float] = {}
        for mat in goods_config.raw_materials:
            lot_qty = lots[mat].quantity if mat in lots else 0
            per_firm = lot_qty / firms
            scarcity[mat] = max(1.0, min(3.0, 10.0 / max(1.0, per_firm)))

        # 3. Aggiornamento offerte d'asta (reattivo se battuti, parsimonioso se vinciamo)
        quotes: dict[str, float] = {}
        for mat, lot in lots.items():
            ratio = self.bid_ratios.get(mat, 1.25)
            wanted = self.last_wanted.get(mat, 0)
            if wanted > 0 and self.identity is not None:
                fill = wins[mat] / wanted
                if fill < 0.85:
                    # Rilancio rapido per non restare a secco contro gli umani
                    ratio = min(6.0, ratio * (1.16 + 0.24 * (1.0 - fill)))
                elif fill >= 0.98:
                    # Vinciamo tutto: scendi gradualmente per non regalare soldi col pay-your-bid
                    ratio = max(1.005, ratio * 0.93)

            self.bid_ratios[mat] = ratio
            nudge = 1.005 if state.turn <= 2 else 1.0
            quotes[mat] = lot.reserve_price * ratio * nudge

        # 4. Valutazione ricette e costi reali
        alternatives = {}
        for name, recipe in goods_config.recipes.items():
            if not recipe.inputs:
                continue
            feasible = min(
                (state.inventory.get(m, 0) + (lots[m].quantity if m in lots else 0)) // n
                for m, n in recipe.inputs.items()
            )
            est_cost = sum(n * quotes.get(m, 0.0) for m, n in recipe.inputs.items())
            base_cost = sum(
                n * (lots[m].reserve_price if m in lots else self.avg_paid_cost.get(m, 10.0))
                for m, n in recipe.inputs.items()
            )
            bottleneck_weight = sum(n * scarcity.get(m, 1.0) for m, n in recipe.inputs.items())
            score = (feasible == 0, est_cost, -feasible, name)
            good = recipe.output_good
            if good not in alternatives or score < alternatives[good][0]:
                alternatives[good] = (
                    score, name, recipe, feasible, est_cost, base_cost, bottleneck_weight
                )

        # 5. Pricing universale (nessun floor rigido + reazione immediata a 0 vendite)
        ranking = []
        for good, (_, name, recipe, feasible, est_cost, base_cost, b_weight) in alternatives.items():
            stock = state.inventory.get(good, 0)
            ref_cost = max(1.0, base_cost)
            # Pavimento puramente economico: costo + 8% margine netto minimo
            true_floor = max(est_cost * 1.08, ref_cost * 1.10)

            if good not in self.anchor_prices:
                # Ancora intermedia: al turno 1-2 esploreremo da 1.30x fino a 7.20x ref_cost
                self.anchor_prices[good] = ref_cost * 2.8

            if good not in self.ema_demand:
                self.ema_demand[good] = 2.0

            anchor = self.anchor_prices[good]
            demand = self.ema_demand[good]
            sold = sales[good]
            had_stock = (stock + sold) > 0

            if self.identity is not None and had_stock:
                if sold > 0:
                    self.dry_turns[good] = 0
                    avg_p = my_rev[good] / sold
                    top_p = my_max_sold[good]

                    obs_demand = sold * 1.35 if stock == 0 else float(sold)
                    demand = 0.55 * demand + 0.45 * max(1.0, obs_demand)

                    if top_p >= anchor * 1.18:
                        # Venduta la fascia alta: spingi subito il prezzo verso il tetto pagato!
                        anchor = max(anchor * 1.15, top_p * 0.98)
                    elif top_p >= anchor * 0.95:
                        # Venduta la fascia centrale: alza gradualmente
                        anchor *= 1.06
                    else:
                        # Venduta solo la fascia bassa: allinea l'ancora vicino al prezzo che il mercato accetta
                        anchor = 0.50 * anchor + 0.50 * max(avg_p * 1.12, true_floor * 1.05)
                else:
                    # Zero vendite con merce disponibile: taglia subito per trovare il WTP dello scenario
                    self.dry_turns[good] += 1
                    anchor *= 0.78
                    demand = max(1.0, demand * 0.80)

            # Segnale dai prezzi visibili di mercato
            if good in market_max and market_max[good] > anchor:
                anchor = 0.60 * anchor + 0.40 * market_max[good]

            # Controllo pressione competitiva tramite public_inventories
            comp_stock = sum(
                inv.get(good, 0) for inv in state.public_inventories.values()
            ) - stock
            if state.turn > 2 and stock > 0:
                if comp_stock == 0:
                    # Gli avversari hanno finito le scorte: sfrutta il monopolio
                    anchor *= 1.08
                elif comp_stock > 15 * max(1, firms - 1) and sold == 0:
                    # Mercato saturo di invenduto altrui: sii più aggressivo sul prezzo
                    anchor *= 0.92

            anchor = max(true_floor, anchor)
            self.anchor_prices[good] = anchor
            self.ema_demand[good] = max(1.0, min(150.0, demand))

            # Tetto massimo di offerta in asta: mai pagare più del 62% del prezzo di vendita stimato
            max_recipe_cost = anchor * 0.62
            if est_cost > max_recipe_cost and est_cost > 0:
                scale = max_recipe_cost / est_cost
                for m in recipe.inputs:
                    if m in lots:
                        quotes[m] = max(lots[m].reserve_price, quotes[m] * scale)
                est_cost = sum(n * quotes.get(m, 0.0) for m, n in recipe.inputs.items())

            margin = anchor - est_cost

            # PROTEZIONE CASSA (Anti-Deadstock):
            # Se il prodotto non ha ancora venduto nemmeno 1 unità, produci max 2 unità di test!
            if self.verified_sales[good] == 0:
                desired_stock = 2
            else:
                # Scala la produzione solo sulla domanda reale verificata (buffer snello 1.35x)
                desired_stock = ceil(self.ema_demand[good] * 1.35)

            goal = max(0, desired_stock - stock)
            target = min(goal, feasible) if margin > 0 else 0

            # Priorità ai prodotti con margine alto per risorsa scarsa e domanda provata
            proven_bonus = 1.5 if self.verified_sales[good] > 0 else 0.7
            priority = (margin / max(0.5, b_weight)) * proven_bonus * (1.0 + 0.2 * min(10.0, self.ema_demand[good]))
            ranking.append((priority, name, good, recipe, target, true_floor, ref_cost))

        # 6. Allocazione Budget Prudente + Assemblaggio + Split Listings
        raw = dict(state.inventory)
        available = {m: lot.quantity for m, lot in lots.items()}
        # Mai spendere più del 45% della cassa in un solo turno (evita di restare al verde con magazzino pieno)
        budget = 0.45 * state.capital

        wanted = Counter()
        assembly: list[AssemblyOrder] = []
        listings: list[Listing] = []

        for _, name, good, recipe, target, true_floor, ref_cost in sorted(ranking, reverse=True):
            if target <= 0:
                quantity = 0
            else:
                quantity = target
                for m, n in recipe.inputs.items():
                    quantity = min(quantity, (raw.get(m, 0) + available.get(m, 0)) // n)

                def spend(units: int) -> float:
                    return sum(
                        max(0, units * n - raw.get(m, 0)) * quotes.get(m, 0.0)
                        for m, n in recipe.inputs.items()
                    )

                if spend(quantity) > budget:
                    low, high = 0, quantity
                    while low < high:
                        mid = (low + high + 1) // 2
                        if spend(mid) <= budget:
                            low = mid
                        else:
                            high = mid - 1
                    quantity = low

                budget -= spend(quantity)

            if quantity > 0:
                assembly.append(AssemblyOrder(name, int(quantity)))
                for m, n in recipe.inputs.items():
                    req = n * quantity
                    to_buy = max(0, req - raw.get(m, 0))
                    wanted[m] += to_buy
                    available[m] = max(0, available.get(m, 0) - to_buy)
                    raw[m] = max(0, raw.get(m, 0) - req)

            listed = state.inventory.get(good, 0) + quantity
            p = self.anchor_prices[good]
            if listed <= 0 or not isfinite(p) or p <= 0:
                continue

            # Se il prodotto non ha ancora venduto nulla, copri un ventaglio larghissimo (1.30x -> 7.20x)
            if self.verified_sales[good] == 0:
                p_low = max(true_floor, ref_cost * 1.30)
                p_mid = max(p_low * 1.15, ref_cost * 2.20)
                p_high = max(p_mid * 1.25, ref_cost * 4.00)
                p_top = max(p_high * 1.35, ref_cost * 7.20)
            else:
                p_low = max(true_floor, p * 0.84)
                p_mid = max(p_low * 1.04, p * 1.00)
                p_high = max(p_mid * 1.06, p * 1.25)
                p_top = max(p_high * 1.08, p * 1.55)

            if listed == 1:
                listings.append(Listing(good, 1, round(p_mid, 2)))
            elif listed == 2:
                listings.append(Listing(good, 1, round(p_high, 2)))
                listings.append(Listing(good, 1, round(p_low, 2)))
            elif listed == 3:
                listings.append(Listing(good, 1, round(p_mid, 2)))
                listings.append(Listing(good, 1, round(p_top, 2)))
                listings.append(Listing(good, 1, round(p_low, 2)))
            else:
                q_mid = max(1, floor(listed * 0.35))
                q_high = max(1, floor(listed * 0.25))
                q_top = max(1, floor(listed * 0.15))
                q_low = max(1, listed - q_mid - q_high - q_top)
                listings.append(Listing(good, q_mid, round(p_mid, 2)))
                listings.append(Listing(good, q_high, round(p_high, 2)))
                listings.append(Listing(good, q_top, round(p_top, 2)))
                listings.append(Listing(good, q_low, round(p_low, 2)))

        self.last_wanted = wanted
        bids = [
            Bid(m, int(qty), round(float(quotes[m]), 2))
            for m, qty in wanted.items()
            if qty > 0 and m in lots and quotes[m] >= lots[m].reserve_price
        ]
        return Decision(bids=bids, assembly=assembly, listings=listings)