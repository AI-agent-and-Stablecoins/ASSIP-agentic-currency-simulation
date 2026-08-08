Here is the complete, hyper-detailed **Phase 3 Master Engineering Specification (`phase_3_instructions.md`)**.

It removes all references to a long-term roadmap (establishing Phase 3 as the complete final release) and incorporates all of your economic, statistical, architectural, memory, network effect, and provenance requirements into one single, production-ready specification document.

---

```markdown
# ASSIP Future of Finance Lab --- Phase 3 Master Engineering Specification: Full-Scale Multi-Agent Engine, Econometric Telemetry & Interactive Dashboard

> **Notice for Claude Code / LLM Developer:** This document serves as the final, authoritative engineering specification for the repository. Every subsystem, interface, experiment, and database schema constructed must strictly conform to the architecture, boundaries, mathematical formulations, and guidelines defined in this document. Phase 3 is the complete and final operational phase of this platform.

---

## 1. Executive Summary & Core Engineering Philosophy

* **Objective:** Construct the complete full-scale simulation platform orchestrator (`src/simulation/matrix_runner.py`), factor-isolation sandbox suite, dynamic economic shock engine, adaptive agent memory and social learning layer, publication-grade econometric analytics suite, raw SQLite telemetry persistence, and real-time interactive Streamlit dashboard (`dashboard/app.py`).
* **Inviolable Invariant (Deterministic Separation):** LLMs act strictly as reasoning, decision-making, and negotiating engines. The core Python/SQLite backend retains 100% deterministic authority over wallet balances, transaction validation, settlement execution, fee subtractions, conversion tax calculations, utility evaluations, memory decay, and ledger immutability.
* **Core Research Question:** What economic, structural, social, and governance characteristics of digital money do autonomous AI agents develop preferences for over time under multi-turn negotiations, network effects, and macroeconomic shocks?

---

## 2. Master Folder Structure & System Scope


```

.
├── configs/
│   ├── agent_profiles.yaml         # Risk parameters, initial wealth, network influence weights
│   ├── currencies.yaml             # Asset attributes, reserves, GENIUS Act status, spread profiles
│   ├── blockchains.yaml            # L1/L2 gas parameters, settlement block speeds, bridge costs
│   ├── scenarios/                  # Shock schedules, factor-isolation parameters
│   └── simulation.yaml             # Run matrix, seeds, timesteps, research toggles
├── database/
│   ├── schema.sql                  # SQLite relational schema definition
│   └── simulation_results.db       # Active SQLite telemetry storage
├── dashboard/
│   ├── app.py                      # Streamlit dashboard main entrypoint
│   └── components/                 # Plotly charts, memory inspector, hypothesis cards
├── metrics/
│   ├── hypothesis_verifier.py      # Regression engines, p-values, 95% CIs, R^2 calculations
│   └── hallucination_detector.py   # True value vs negotiated value telemetry
├── src/
│   ├── agents/
│   │   ├── base_agent.py           # Core agent class, wallet state, purchasing power math
│   │   ├── memory.py               # Episodic and semantic memory store, learning loop
│   │   └── profiles.py             # Bank, Consumer, Investor, Merchant, Regulator agents
│   ├── social/
│   │   └── network_effects.py      # Peer observation, merchant cascades, adoption ratio
│   ├── utility/
│   │   └── utility_functions.py    # CRRA, CARA, Epstein-Zin, Multi-Attribute formulations
│   ├── blockchain/                 # Gas pricing engines, block settlement, bridge rails
│   ├── currencies/                 # Stablecoins, gold tokens, tokenized deposits
│   ├── economy/
│   │   └── shocks.py               # Inflation, bank failures, depegs, fee spikes, FX shocks
│   ├── governance/                 # GENIUS Act compliance, audit rating engines
│   ├── llm/
│   │   ├── llm_router.py           # OpenRouter API wrapper, retries, fallback logic
│   │   ├── agent_reasoning.py      # Prompt construction, memory formatting, schema output
│   │   └── prompts/                # Role-specific system and user prompts
│   ├── market/                     # Order books, spreads, pricing reference engines
│   ├── negotiation/                # Multi-turn state machine, negotiation logs
│   ├── simulation/
│   │   ├── timestep.py             # Single timestep lifecycle algorithm
│   │   └── matrix_runner.py        # Master orchestrator for full factor matrix runs
│   └── transactions/
│       └── validation.py           # Deterministic validation, settlement, ledger writes
├── tests/                          # Pytest suite covering all modules
└── project_instructions.md         # Master context file

```

---

## 3. Agent Architecture, Utility Mathematics & Adaptive Learning

### A. Real Purchasing Power Utility Model
Agents prioritize **Real Purchasing Power** ($W_{\text{real}}$) over nominal currency balances to evaluate true economic utility in inflation-prone environments:

$$W_{\text{real}, i, t} = \frac{\sum_{k} S_{i, k, t} \cdot P_{k, t}}{I_{\text{price}, t}}$$

Where $S_{i, k, t}$ is agent $i$'s balance of currency $k$, $P_{k, t}$ is currency $k$'s exchange rate to USD, and $I_{\text{price}, t}$ is the current price index of the agent's native economic zone.

### B. Mathematical Utility Formulations (`src/utility/utility_functions.py`)
Agents evaluate choices using one of five explicit utility specifications:

1. **Risk Neutral:**
   $$U(c) = c$$

2. **Constant Relative Risk Aversion (CRRA):**
   $$U(c) = \begin{cases} \frac{c^{1-\sigma} - 1}{1 - \sigma}, & \text{if } \sigma \neq 1 \\ \ln(c), & \text{if } \sigma = 1 \end{cases}$$
   * *Dynamic Risk Parameter:* $\sigma_{i, t}$ adjusts adaptively based on loss history.

3. **Constant Absolute Risk Aversion (CARA):**
   $$U(c) = -e^{-\alpha c}$$

4. **Epstein-Zin Recursive Utility:**
   * Separates risk aversion ($\sigma$) from intertemporal elasticity of substitution ($\psi$):
     $$V_t = \left[ (1-\beta) c_t^{1 - 1/\psi} + \beta \left( \mathbb{E}_t [V_{t+1}^{1-\sigma}] \right)^{\frac{1 - 1/\psi}{1-\sigma}} \right]^{\frac{1}{1 - 1/\psi}}$$

5. **Multi-Attribute Utility with Currency Network Effects:**
   $$U_{i, k, j} = w_{\text{gov}} \cdot \text{Gov}_k + w_{\text{liq}} \cdot (1 - \text{Spread}_k) + w_{\text{net}} \cdot \text{AcceptanceRate}_{k, t} + w_{\text{stab}} \cdot (1 - \text{PegError}_k) + w_{\text{priv}} \cdot \text{Priv}_k - w_{\text{fee}} \cdot \text{GasFee}_{j,k} - w_{\text{vol}} \cdot \text{Vol}_k$$

### C. Social Influence & Currency Network Effects (`src/social/network_effects.py`)
* **Peer Observation:** Agents observe currency choices within their local network graph or demographic peer group.
* **Merchant Acceptance Cascades:** Threshold-based dynamic adoption:
  $$\text{If } \frac{\sum_{m \in M} \mathbb{I}(m \text{ accepts currency } k)}{|M|} \ge \Theta_{\text{cascade}} \implies \text{Adoption Cascade Triggered}$$
  When total merchant acceptance crosses the threshold $\Theta_{\text{cascade}}$ (e.g., $70\%$), non-accepting merchants experience accelerated utility weighting toward currency $k$, creating **winner-take-all network dynamics**.

### D. Bounded Agent Memory System (`src/agents/memory.py`)
Agents store episodic events and semantic historical knowledge. Prompts assemble memory context dynamically into the system prompt:

```json
{
  "historical_memory": [
    "USDT depegged twice in the past 30 timesteps.",
    "Buyer 18 negotiated aggressively and defaulted on offer.",
    "Ethereum gas exploded to 180 Gwei in timestep 391.",
    "Bank Alpha defaulted during liquidity crunch.",
    "USDC is currently accepted by 97% of local merchants."
  ]
}

```

### E. Adaptive Learning & Dynamic Preference Adjustment

Agents adapt their parameters based on market feedback:

* **Loss-Driven Risk Aversion Scaling:**

$$\sigma_{i, t+1} = \min\left(\sigma_{\max}, \, \sigma_{i, t} + \eta_{\text{risk}} \cdot \frac{\text{Loss}_{i, t}}{W_{\text{real}, i, t}}\right)$$



If an agent suffers capital losses (e.g., during a depegging event), their risk aversion parameter $\sigma$ increases, accelerating their migration toward safer, compliant assets (USDC / PAXG).
* **Friction-Driven Rail Adaptation:** Sustained high gas fees on a chain (e.g., Ethereum L1) permanently increase the agent's gas penalty weight $w_{\text{fee}}$, driving routing migration toward L2s (Base, Arbitrum) or Solana.

---

## 4. Dual Research Modes: Factual Context vs. Autonomous Agent Self-Research

1. **Factual Knowledge Mode:** Exact macro parameters, historical peg variance tables, and verified issuer audit scores are directly injected into the LLM system prompt.
2. **Autonomous Agent Self-Research Mode:** Agents are given tool-calling privileges (querying local vector databases, pre-indexed news archives, or web search APIs) to discover reserve audit logs, regulatory updates (e.g., GENIUS Act compliance), and issuer balance sheets prior to formulating proposals.
3. **Comparative Analysis Metrics:** The system automatically calculates and reports statistical divergence between Factual and Self-Research modes across:
* Price discovery accuracy and negotiation length
* Susceptibility to LLM hallucinations
* Trajectory of preference stability under economic uncertainty



---

## 5. Experiment Execution Matrix & Factor Isolation Sandboxes

### A. The Master Simulation ("One Big Simulation")

A full-scale environment executing all agent types across all supported assets and blockchains simultaneously:

* **Currencies:** USD (USDC, USDT, FDUSD, DAI), EUR (EURC, EURT), Gold (PAXG, XAUT), Bank (Tokenized Deposits).
* **Blockchains:** Ethereum, Arbitrum, Base, Solana.
* **Active Mechanisms:** All economic shocks, multi-turn negotiations, network effects, and adaptive memory active concurrently.

### B. Pairwise Factor Isolation Sandboxes (`experiments/sandboxes/`)

Controlled sandbox scripts isolating individual economic trade-offs by holding all other variables constant:

| Sandbox Scenario | Option A Attributes | Option B Attributes | Target Decision Trait |
| --- | --- | --- | --- |
| **1. Liquidity vs. Governance** | USDT (Spread: 0.01%, Non-compliant) | USDC (Spread: 0.10%, Compliant) | Governance premium threshold |
| **2. Governance vs. Stability** | Compliant coin ($\text{PegError} = 0.02$) | Non-compliant coin ($\text{PegError} = 0.00$) | Regulatory vs peg priority |
| **3. Liquidity vs. Stability** | Low spread (0.01%), high variance ($\pm \$0.04$) | High spread (0.25%), exact peg ($\$1.000$) | Spread vs volatility tolerance |
| **4. Asset Backing vs. Liquidity** | Gold token PAXG (Spread: 0.30%) | Fiat stablecoin USDC (Spread: 0.01%) | Hard asset preference |
| **5. Asset Backing vs. Stability** | Gold token XAUT (Commodity volatility) | Tokenized Deposit (Fiat stability) | Inflation hedge vs fiat peg |
| **6. Asset Backing vs. Governance** | Bank Tokenized Deposit (Bank credit risk) | DAI (Crypto-collateralized / Algorithmic) | Centralized vs decentralized trust |
| **7. Privacy vs. Friction** | USDCx / Aleo Rail (5% spread/fee penalty) | Public Transparent Rail (0.01% fee) | Anonymity willingness-to-pay |

### C. Dynamic Economic Shock Engine (`src/economy/shocks.py`)

Dynamic event triggers injected during simulation execution:

* `inflation_shock`: Surges local inflation rate ($I_{\text{price}}$), driving flight to gold (PAXG) or yield.
* `bank_failure_shock`: Drops market confidence index, initiating runs on tokenized deposits.
* `depegging_event`: Forces target asset `peg_error` to spike (e.g., USDT drops to $\$0.91$).
* `fee_spike`: Multiplies Ethereum L1 gas fees by 10x–50x, testing L2/Solana routing migration.
* `fx_volatility_shock`: Spikes EUR/USD exchange volatility during cross-border runs.

### D. Cross-Border Settlement Framework

* **Pairings:** US-based agents (optimizing real USD purchasing power) transacting with EU-based agents (optimizing real EUR purchasing power).
* **Conversion Friction:** Non-native settlement incurs a configurable conversion tax percentage ($\tau_{\text{fx}}$).
* **Volatility Effects:** Measures how fluctuations in the EUR/USD exchange rate shift preference toward USD stablecoins as a global settlement unit.

---

## 6. Experiment Reproducibility, Provenance & Intervention Logging

### A. Provenance Metadata (Saved to DB per Run)

To guarantee strict scientific reproducibility, every simulation run automatically captures and records:

* `random_seed`: Integer seed governing all stochastic code paths.
* `model_version` & `openrouter_model_id`: Exact LLM identifier used.
* `prompt_version_hash`: SHA-256 hash of all system and prompt templates.
* `git_commit_hash`: Git SHA commit hash at time of execution.
* `timestamp`: UTC execution timestamp.
* `config_hash`: SHA-256 hash of resolved YAML configuration files.

### B. Step-Indexed Intervention Logging (`InterventionLog`)

Every macroeconomic shock or environment change is logged with its precise step index:

```
[Step 0212] INTERVENTION LOGGED: Inflation Shock Injected (Inflation Rate -> 8.5%)
[Step 0391] INTERVENTION LOGGED: Gas Spike Injected (Ethereum L1 Gas -> 180 Gwei)
[Step 0610] INTERVENTION LOGGED: USDT Depeg Injected (Peg Value -> $0.92)
[Step 0822] INTERVENTION LOGGED: Regional Bank Failure Shock (Confidence Index -> 0.35)

```

---

## 7. Publication-Grade Econometric Analytics Engine (`metrics/`)

The analytics engine (`src/metrics/hypothesis_verifier.py`) evaluates hypotheses using formal econometric methods, outputting publication-grade metrics rather than basic binary flags:

### Required Statistical Outputs per Hypothesis:

* **Estimated Coefficient ($\beta$):** Impact magnitude and direction.
* **Standard Error ($\text{SE}$):** Estimator precision.
* **95% Confidence Interval ($[\beta_{\text{lower}}, \beta_{\text{upper}}]$):** Rigorous confidence bounds.
* **$p$-value:** Statistical significance value.
* **Goodness-of-Fit ($R^2$ & Adjusted $R^2$):** Variance explanation score.

### Target Hypotheses to Evaluate:

1. **H1 (Risk Aversion vs. Currency Choice):** Higher CRRA $\sigma \implies$ statistically significant shift favoring USD stablecoins over EUR stablecoins.
2. **H2 (Risk Aversion vs. Liquidity & Fees):** Higher CRRA $\sigma \implies$ prioritizing low bid-ask spreads over low gas fee concerns.
3. **H3 (Risk Aversion vs. Governance):** Higher CRRA $\sigma \implies$ prioritizing GENIUS Act compliance over market liquidity.
4. **H4 (Crisis Proximity vs. Gold Backing):** Higher perceived banking crisis / depeg probability $\implies$ shift toward gold-backed tokens (PAXG/XAUT).
5. **H5 (Cross-Border Volatility):** Higher exchange rate volatility $\implies$ increased preference for USD stablecoins in cross-border settlement.
6. **H6 (Privacy Premium Threshold):** Quantifies the exact maximum spread/fee penalty agents tolerate to preserve transaction privacy.

---

## 8. Raw Data Persistence Layer (SQLite Database Schema)

All telemetry persists to SQLite database tables (`database/simulation_results.db`) without pre-aggregation:

```sql
-- Run Provenance & Metadata
CREATE TABLE simulation_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    scenario_name VARCHAR(128) NOT NULL,
    research_mode VARCHAR(32) NOT NULL, -- 'Factual' or 'Self-Research'
    random_seed INT NOT NULL,
    openrouter_model_id VARCHAR(128) NOT NULL,
    prompt_version_hash VARCHAR(64) NOT NULL,
    git_commit_hash VARCHAR(64) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step-Indexed Intervention Events
CREATE TABLE intervention_logs (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id VARCHAR(64) NOT NULL,
    timestep INT NOT NULL,
    shock_type VARCHAR(64) NOT NULL,
    target_variable VARCHAR(64) NOT NULL,
    magnitude FLOAT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES simulation_runs(run_id)
);

-- Daily Macro State Telemetry
CREATE TABLE timestep_logs (
    run_id VARCHAR(64) NOT NULL,
    timestep INT NOT NULL,
    inflation_rate FLOAT NOT NULL,
    confidence_index FLOAT NOT NULL,
    eth_gas_fee_gwei FLOAT NOT NULL,
    solana_gas_fee_usd FLOAT NOT NULL,
    eur_usd_exchange_rate FLOAT NOT NULL,
    PRIMARY KEY(run_id, timestep)
);

-- Agent State & Memory Logs
CREATE TABLE agent_states (
    run_id VARCHAR(64) NOT NULL,
    timestep INT NOT NULL,
    agent_id VARCHAR(64) NOT NULL,
    risk_profile VARCHAR(32) NOT NULL,
    crra_sigma FLOAT NOT NULL,
    real_purchasing_power FLOAT NOT NULL,
    usd_balance FLOAT NOT NULL,
    eur_balance FLOAT NOT NULL,
    gold_balance FLOAT NOT NULL,
    utility_score FLOAT NOT NULL,
    PRIMARY KEY(run_id, timestep, agent_id)
);

-- Agent Episodic Memory Records
CREATE TABLE agent_memory_logs (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id VARCHAR(64) NOT NULL,
    timestep INT NOT NULL,
    agent_id VARCHAR(64) NOT NULL,
    memory_type VARCHAR(32) NOT NULL, -- 'Depeg', 'Default', 'GasSpike', 'Network'
    memory_text TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES simulation_runs(run_id)
);

-- Raw Negotiation Transcripts
CREATE TABLE negotiation_raw_logs (
    negotiation_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    timestep INT NOT NULL,
    turn_index INT NOT NULL,
    buyer_id VARCHAR(64) NOT NULL,
    seller_id VARCHAR(64) NOT NULL,
    system_prompt TEXT NOT NULL,
    llm_reasoning TEXT NOT NULL,
    raw_json_offer TEXT NOT NULL,
    status VARCHAR(32) NOT NULL, -- 'Accepted', 'Rejected', 'Countered'
    FOREIGN KEY(run_id) REFERENCES simulation_runs(run_id)
);

-- Deterministic Settlement Ledger
CREATE TABLE transactions_ledger (
    transaction_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    timestep INT NOT NULL,
    chain VARCHAR(32) NOT NULL,
    currency VARCHAR(32) NOT NULL,
    gross_amount FLOAT NOT NULL,
    fx_tax_paid FLOAT NOT NULL,
    gas_paid FLOAT NOT NULL,
    settlement_status VARCHAR(32) NOT NULL,
    FOREIGN KEY(run_id) REFERENCES simulation_runs(run_id)
);

-- Hallucination Measurement Telemetry
CREATE TABLE hallucination_telemetry (
    telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64) NOT NULL,
    expected_fair_value FLOAT NOT NULL,
    llm_negotiated_value FLOAT NOT NULL,
    error_magnitude_pct FLOAT NOT NULL,
    is_hallucination BOOLEAN NOT NULL,
    FOREIGN KEY(transaction_id) REFERENCES transactions_ledger(transaction_id)
);

```

---

## 9. Interactive Streamlit Monitoring Dashboard (`dashboard/app.py`)

The platform features a real-time interactive Streamlit web application (`streamlit run dashboard/app.py`):

### View Modules:

1. **Control Center:** Launch Master Runs or Factor Isolation Sandboxes, adjust baseline agent parameters ($\sigma$), trigger live economic shocks, toggle Factual vs. Self-Research modes, and view provenance metadata (seed, git hash).
2. **Macro & Adoption Analytics:** Interactive Plotly charts displaying stablecoin market share adoption cascades, blockchain gas fee trends, network effect tipping curves, and real purchasing power velocity.
3. **Agent & Memory Deep-Dive:** Inspector panel for individual agents displaying real-time wallet balances, utility curves, dynamically updating risk aversion ($\sigma_{i, t}$), and full historical episodic memory logs.
4. **Negotiation Explorer & Telemetry:** Filterable database log viewer displaying complete LLM system prompts, raw chain-of-thought reasoning strings, structured JSON proposals, and hallucination overpayment scatter plots.
5. **Publication-Grade Econometric Board:** Live summary cards displaying regression outputs for Hypotheses H1 through H6, complete with **Coefficients ($\beta$), Standard Errors ($\text{SE}$), 95% Confidence Intervals, $p$-values, and $R^2$ values**.

---

## 10. Complete Timestep Lifecycle Algorithm (`simulation_runner.py`)

For each timestep $t = 1 \dots T$:

1. **Apply Macro Environment & Interventions:** Update inflation rates, gas fees, and FX rates. Process scheduled economic shocks and append to `intervention_logs`.
2. **Compute Network Effects:** Calculate updated merchant acceptance ratios ($\text{AcceptanceRate}_{k, t}$) and trigger network cascades if thresholds are crossed.
3. **Update Adaptive Memory & Parameters:** Adjust agent risk aversion parameters ($\sigma_{i, t}$) and gas penalties based on prior timestep losses or high gas friction.
4. **Schedule Transacting Pairs:** Match buyer, seller, merchant, and bank agents.
5. **Agent Observation & Research Step:**
* Construct observation payload with real purchasing power balances, market spreads, and relevant memory items.
* If in *Self-Research Mode*, execute tool-calling research queries over reserve audits and news archives.


6. **Utility Evaluation:** Compute multi-attribute real purchasing power utility across potential transaction paths.
7. **LLM Reasoning & Multi-Turn Negotiation:**
* Pass prompt payload to OpenRouter API (`src/llm/llm_router.py`).
* LLM generates chain-of-thought reasoning and structured JSON decision output (`Action`, `Currency`, `Chain`, `Price`).
* Execute counter-offer loops until accepted, rejected, or max rounds reached.


8. **Deterministic Backend Validation:** Validate wallet solvency, asset availability, compliance standing, and gas balance (`src/transactions/validation.py`).
9. **Settlement & Ledger Update:** Transfer balances deterministically, subtract gas fees and conversion taxes ($\tau_{\text{fx}}$), and record to `transactions_ledger`.
10. **Hallucination Telemetry Scoring:** Compare agreed transaction value against deterministic valuation engine; log discrepancies to `hallucination_telemetry`.
11. **Telemetry & DB Persistence:** Write complete unaggregated raw step state to SQLite database.

---

## 11. Coding Rules & System Invariants

* **Strict Type Annotations:** All Python functions must include explicit type hints.
* **Pydantic v2 Models:** Enforce strict schema validation across configuration files, API payloads, and LLM structured JSON responses.
* **Zero Global State:** All system dependencies must be explicitly injected via constructors or configuration instances.
* **Comprehensive Pytest Suite:** Every module must have corresponding unit/integration tests in `tests/` validating balance conservation, utility accuracy, memory decay math, and deterministic validation rules.
* **Inviolable Balance Invariant:** LLMs cannot directly write to or alter wallet states. State mutations occur exclusively via deterministic validation logic in `src/transactions/validation.py`.

```

***

### You are all set! 
You can now copy the code block above, paste it into `phase_3_instructions.md` (or save it as `project_instructions.md`), and use it to execute the build.

```