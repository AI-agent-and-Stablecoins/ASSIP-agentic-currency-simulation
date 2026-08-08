\# Extending the Simulation: Shocks, Trust Score, and Historical Context

This is a design spec, not yet implemented. It covers what's needed to test  
the three hypotheses properly \*inside the live simulation\* rather than  
through the static utility-weight proxy used previously. Grounded in the  
actual codebase: \`src/economy/shocks.py\` currently defines exactly \*\*4\*\*  
shock types (\`inflation\`, \`bank\_failure\`, \`gold\_rally\`, \`fee\_spike\`), and  
none of them touches a currency's own \`peg\_error\` — only aggregate  
\`macro\_state\` fields (\`inflation\`, \`confidence\_index\`, \`gold\_price\`). That's  
the central gap this spec addresses.

\---

\#\# 1\. Full list of economic shocks needed

\#\#\# 1.1 Existing shocks (already implemented, keep as-is)

| Shock | Mechanism | Field(s) touched |  
|---|---|---|  
| \`inflation\` | Adds to inflation rate | \`macro\_state.inflation\` |  
| \`bank\_failure\` | Drops aggregate confidence | \`macro\_state.confidence\_index\` |  
| \`gold\_rally\` | Multiplies gold price | \`macro\_state.gold\_price\`, \`peg\_reference\_rates\["XAU"\]\` |  
| \`fee\_spike\` | Applied at the blockchain-config layer, not macro state | gas fees |

None of these can move a \*specific currency's\* \`peg\_error\`, target a  
\*specific issuer\*, or model an FX move between two currency blocs (USD vs.  
EUR). All three new hypotheses need at least one of those.

\#\#\# 1.2 New shocks needed, by hypothesis

\*\*H1 — risk aversion → governance/compliance over liquidity\*\*

| Shock | Description | Mechanism |  
|---|---|---|  
| \`regulatory\_enforcement\` | GENIUS Act enforcement action against a named non-compliant issuer (fine, redemption freeze) | Spikes \`issuer\_risk\` for the target currency; temporarily drops its \`liquidity\_score\` (frozen redemptions create real friction, not just reputational risk) |  
| \`liquidity\_crunch\` | A market maker withdraws from one currency, independent of governance | Temporarily drops \`liquidity\_score\` for the target currency only |  
| \`governance\_downgrade\` | A reserve-audit failure or transparency scandal, short of a full depeg | Drops \`governance\_score\` for the target currency (currently static in config — this shock needs to make it dynamic) |

\`liquidity\_crunch\` and \`regulatory\_enforcement\` are the pair that actually  
puts governance and liquidity in tension \*live\* — right now that tension  
only exists as a hand-picked config difference (USDT vs. TDUSD), never as  
something that happens \*during\* a run.

\*\*H2 — crisis proximity → gold-backed preference\*\*

| Shock | Description | Mechanism |  
|---|---|---|  
| \`depeg\_event\` | A specific stablecoin's price deviates sharply from its peg, then mean-reverts over several days | Spikes \`peg\_error\` for the target currency, decaying back to baseline over a configurable number of days (unlike \`bank\_failure\`, which steps \`confidence\_index\` down permanently) |  
| \`crisis\_warning\` | An earlier, smaller signal preceding a scheduled \`bank\_failure\`/\`depeg\_event\` by N days (a rumor, a rating downgrade, a regulator statement) | Small \`confidence\_index\` dip, or a dedicated \`warning\_level\` field (see §2) |  
| \`bank\_failure\` \*(extend, don't replace)\* | Add an optional \`target\_issuer\` field so a failure can hit one issuer's currencies specifically, enabling contagion tests | \`confidence\_index\` (aggregate) \+ issuer-specific \`trust\_score\` hit (§2) |

\`crisis\_warning\` is what actually lets you test \*proximity\* as a variable —  
run the same eventual \`depeg\_event\`/\`bank\_failure\` but vary how many days  
of advance warning precede it (0, 5, 10, 20), and measure gold-backed share  
in the days between the warning and the event.

\*\*H3 — cross-border volatility → USD over EUR stablecoins\*\*

| Shock | Description | Mechanism |  
|---|---|---|  
| \`fx\_volatility\_shock\` | A Euro-area-specific shock (political crisis, ECB surprise) that raises volatility only for EUR-pegged currencies | Spikes \`peg\_error\` for all currencies with \`peg \== "EUR"\`, holding USD-pegged currencies fixed |  
| \`fx\_rate\_shock\` | A spot move in the EUR/USD cross-rate itself, distinct from any single stablecoin's peg deviation | Shocks \`macro\_state.peg\_reference\_rates\["EUR"\]\` directly |  
| \`capital\_controls\` | Regulatory friction on cross-border EUR redemption | Raises effective redemption friction / lowers effective liquidity for EUR-pegged currencies in cross-border transactions specifically, separating the "volatility" channel from the "friction" channel |

Splitting \`fx\_volatility\_shock\` (peg\_error) from \`fx\_rate\_shock\` (reference  
rate) matters because the hypothesis says "relative volatility," which  
could mean either a stablecoin \*failing to hold its peg\* or the underlying  
\*fiat currencies moving against each other\* — those are different economic  
events and probably produce different agent reactions.

\#\#\# 1.3 Full combined list (10 shock types)

| \# | Shock | Status | Hypothesis |  
|---|---|---|---|  
| 1 | \`inflation\` | existing | general |  
| 2 | \`bank\_failure\` | existing, extend with \`target\_issuer\` | H2 |  
| 3 | \`gold\_rally\` | existing | H2 (context) |  
| 4 | \`fee\_spike\` | existing | general |  
| 5 | \`regulatory\_enforcement\` | new | H1 |  
| 6 | \`liquidity\_crunch\` | new | H1 |  
| 7 | \`governance\_downgrade\` | new | H1 |  
| 8 | \`depeg\_event\` | new | H2 |  
| 9 | \`crisis\_warning\` | new | H2 |  
| 10 | \`fx\_volatility\_shock\` | new | H3 |  
| 11 | \`fx\_rate\_shock\` | new | H3 |  
| 12 | \`capital\_controls\` | new | H3 |

(12 total, 4 existing \+ 8 new — every new one needs a \`target\_currency\` or  
\`target\_issuer\` field, since none of them are aggregate macro moves.)

\---

\#\# 2\. Trust score: what it is and how it should update

\#\#\# 2.1 Why this is a separate thing from \`governance\_score\`

Today, \`governance\_score\` (in \`configs/currencies/\*.yaml\`) is a \*\*static\*\*  
config value — a fixed, structural judgment about an issuer's institutional  
quality. It never changes during a run. That's fine as a prior, but it  
can't represent \*lived experience\*: an agent that just watched USDT wobble  
should trust it less than an agent reading about USDT for the first time,  
even though \`governance\_score\` hasn't moved.

\`trust\_score\` is the missing \*\*dynamic\*\* counterpart: it starts at the  
currency's \`governance\_score\` and moves up or down based on what actually  
happens during the simulation, then slowly reverts toward the structural  
baseline if nothing else happens. This is also the principled way to get  
H2's "crisis proximity" — instead of the hand-tuned \`crisis\_proximity\`  
bonus I used in the utility-proxy version, \`trust\_score\`'s own decay curve  
\*is\* the proximity signal.

\#\#\# 2.2 Update formula

Let \`τ\_c(t)\` \= trust score for currency \`c\` at day \`t\`, bounded to \`\[0, 1\]\`.

\*\*Initialization:\*\* \`τ\_c(0) \= governance\_score\_c\`

\*\*Quiet day (no event touching \`c\`):\*\* slow mean-reversion toward the  
structural baseline —

\`\`\`  
τ\_c(t) \= τ\_c(t-1) \+ λ\_recover · (governance\_score\_c − τ\_c(t-1))  
\`\`\`

\*\*Event day\*\* — a shock with severity \`s ∈ \[0, 1\]\` hits currency \`c\`  
(\`s\` \= the shock's \`magnitude\`, or \`min(1, peg\_error\_spike / cap)\` for  
\`depeg\_event\`) —

\`\`\`  
τ\_c(t) \= τ\_c(t-1) − λ\_shock · s · τ\_c(t-1)      (floored at 0\)  
\`\`\`

\*\*Contagion\*\* — every other currency \`c'\` sharing the same \`asset\_class\`  
or issuer family as \`c\` takes a smaller hit the same day:

\`\`\`  
τ\_c'(t) \= τ\_c'(t-1) − λ\_contagion · s · τ\_c'(t-1)  
\`\`\`

\*\*Asymmetry (trust crashes fast, rebuilds slowly):\*\*  
\`λ\_shock ≫ λ\_recover\` — e.g. \`λ\_shock \= 0.5\`, \`λ\_recover \= 0.03\`,  
\`λ\_contagion \= 0.2 · λ\_shock\`. These three constants are the whole  
mechanism; they belong in a config file (e.g.  
\`configs/economy/trust\_params.yaml\`), never hardcoded, so they can be  
swept like anything else.

\#\#\# 2.3 Agent-specific perception (where risk aversion plugs in)

Every agent sees the same objective \`τ\_c(t)\`, but risk-averse agents should  
react more to \*how choppy\* it's been recently, not just its current level —  
this is what makes H1's risk-aversion sweep meaningful against a live  
trust signal instead of a static tradeoff pair:

\`\`\`  
perceived\_trust\_i,c(t) \= τ\_c(t) − r\_i · stdev(τ\_c, last W days)  
\`\`\`

where \`r\_i\` is agent \`i\`'s risk-aversion parameter and \`W\` is a rolling  
window (e.g. 30 days). A risk-seeking agent (\`r\_i ≈ 0\`) just reads the  
current trust level; a highly risk-averse agent effectively penalizes  
volatility in trust itself, even if the average level is unchanged.

\#\#\# 2.4 Where it lives in the codebase

\- New module: \`src/economy/trust.py\` — \`TrustLedger\` class holding  
  \`τ\_c(t)\` per currency, with \`update()\` called once per timestep from  
  \`src/simulation/timestep.py\`, fed by whichever shocks fired that day.  
\- New config: \`configs/economy/trust\_params.yaml\` — \`lambda\_shock\`,  
  \`lambda\_recover\`, \`lambda\_contagion\`, rolling window \`W\`.  
\- \`TrustLedger.history(currency, days)\` is the read API that both the  
  utility functions (§2.3) and the LLM prompt context (§3) pull from.

\---

\#\# 3\. Historical (not just factual) information agents should access

\#\#\# 3.1 What's missing today

\`AgentReasoning.build\_prompt()\` and \`AgentObservation\` currently only carry  
\*\*point-in-time snapshot facts\*\*: current wallet balances, each currency's  
current \`governance\_score\`/\`liquidity\_score\`/\`peg\_error\`, the current  
observed price, and the current macro state. There is no trajectory, no  
event log, and no sense of "how did we get here" — an agent reasoning  
today about USDT has no way to know whether USDT has been rock-solid for  
90 days or just wobbled twice last week. That distinction is exactly what  
H2 depends on, and it's currently structurally impossible for the agent to  
know.

\#\#\# 3.2 New fields to add

Extend \`AgentObservation\` (\`src/llm/agent\_reasoning.py\`) with a \`history\`  
section, populated from \`TrustLedger\` and a new lightweight event log  
(\`src/economy/event\_log.py\`, just an append-only list of  
\`{day, shock\_type, target, severity}\` records that \`timestep.py\` writes to  
whenever \`apply\_shock\` fires):

\`\`\`python  
class CurrencyHistory(BaseModel):  
    trust\_now: float  
    trust\_30d\_ago: float  
    trust\_min\_90d: float  
    trend: str                    \# "declining" | "stable" | "recovering"  
    depeg\_events\_90d: int  
    last\_event\_days\_ago: int | None  
    recent\_events: list\[str\]      \# last 3, short human-readable descriptions

class MacroHistory(BaseModel):  
    confidence\_now: float  
    confidence\_30d\_ago: float  
    days\_since\_last\_shock: int | None  
    last\_shock\_type: str | None  
\`\`\`

...added to \`AgentObservation\` alongside the existing current-state fields,  
and rendered into the prompt as a distinct "History" section so the LLM  
can explicitly weigh "what's true right now" against "what's been  
happening" — the same distinction a human trader would make.

Also extend the agent's own \`self.memory\` (already exists in  
\`base\_agent.py\`, currently just past transaction outcomes) to record  
crisis-relevant personal experience: \*"On day 5 I held USDC through a  
banking crisis and lost nothing"\* vs. \*"On day 12 I was mid-transaction in  
USDT when it depegged 8%."\* Same mechanism, just fed a wider set of events.

\#\#\# 3.3 Example: before vs. after

\*\*Before (current, factual-only):\*\*  
\`\`\`  
\- USDT: {"governance\_score": 0.55, "liquidity\_score": 0.98, "peg\_error": 0.0008}  
\`\`\`

\*\*After (with history):\*\*  
\`\`\`  
\- USDT: {"governance\_score": 0.55, "liquidity\_score": 0.98, "peg\_error": 0.0008,  
         "trust\_now": 0.41, "trust\_30d\_ago": 0.55, "trend": "declining",  
         "depeg\_events\_90d": 2, "last\_event\_days\_ago": 6,  
         "recent\_events": \["Day 44: brief 1.8% depeg, recovered in 2 days",  
                            "Day 51: regulatory enforcement action (fine)"\]}  
\`\`\`

The second version is what actually lets an LLM distinguish "structurally  
weaker but nothing's happened lately" from "structurally weaker and  
actively deteriorating right now" — which is the whole point of H2, and  
meaningfully sharpens H1 and H3 too.

\#\#\# 3.4 Data sources this requires (summary)

| Source | New? | Feeds |  
|---|---|---|  
| \`TrustLedger\` (§2.4) | new | \`trust\_now\`, \`trust\_30d\_ago\`, \`trend\`, \`stdev\` for §2.3 |  
| Event log (\`src/economy/event\_log.py\`) | new | \`depeg\_events\_90d\`, \`recent\_events\`, \`last\_event\_days\_ago\` |  
| \`macro\_state\` history buffer | new (currently only holds the \*current\* state, not a time series) | \`MacroHistory\` |  
| \`agent.memory\` | existing, extend | personal experience narrative |

\---

\#\# Implementation checklist

1\. \`src/economy/trust.py\` — \`TrustLedger\`, update formula from §2.2  
2\. \`configs/economy/trust\_params.yaml\` — λ constants, rolling window  
3\. \`src/economy/event\_log.py\` — append-only shock/event record  
4\. Extend \`ShockType\` enum \+ \`apply\_shock()\` in \`src/economy/shocks.py\`  
   with the 8 new shock types from §1.2, each taking a \`target\_currency\`  
   or \`target\_issuer\` field  
5\. 8 new/extended scenario configs under \`configs/scenarios/\` exercising  
   each new shock (plus a \`crisis\_warning\` \+ \`depeg\_event\` pair with a  
   variable gap, for the proximity sweep)  
6\. Extend \`AgentObservation\` \+ \`build\_prompt()\` in  
   \`src/llm/agent\_reasoning.py\` with \`CurrencyHistory\`/\`MacroHistory\`  
7\. Wire \`TrustLedger.update()\` into \`src/simulation/timestep.py\`'s daily loop

CRRA function: set sigma for each agent (if agent is risk neutral then sigma \= 0, if  
the agent is risk averse= 1.5  
And then tell the agent that it wants to maximize U(c) where c is final wealth

In the sandbox it would be nice to isolate a factor: Meaning, we give the agents one  
coin that has better liquidity (lower bid-ask spread) and give them another coin with  
better governance and see which wins out when they start transacting  
\- Isolate liquidity v governance  
\- Governance v stability (closeness to Peg)  
\- Liquidity v stability (closeness to Peg)  
\- Asset backing v liquidity  
\- Asset backing v stability  
\- Asset backing v governance

Cross border – how do we model agents on different sides of the Atlantic (one that  
cares about Euros and one that cares about USD)  
\- One agent maxes wealth in euros, one maxes wealth in USD, each must pay a  
transaction cost to move their money from USDC to Euro (or from EURC to USD)  
– small % tax to convert money if the money is not the type that they want

Cross Border repeat analysis above  
\- Isolate liquidity v governance  
\- Governance v stability (closeness to Peg)  
\- Liquidity v stability (closeness to Peg)  
\- Asset backing v liquidity  
\- Asset backing v stability  
\- Asset backing v governance  
