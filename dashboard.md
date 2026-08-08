













Fri, Jun 19 at 2:31 PM
Mon, Jun 22 at 5:43 PM
Fri, Jun 26 at 12:43 PM
Fri, Jun 26 at 2:47 PM
Fri, Jul 3 at 3:16 PM
Yesterday 11:06 AM

Agentic AI and Stablecoins Ideas(1).pdf
PDF
Please synthesize everything we have discussed about this coding project so far—including all requirements, structural details, and the information from the slideshow I provided—into a single, comprehensive markdown (.md) file. 

This markdown file will be used to initialize a session with Claude, serving as its master instructions and context document.

Please use the following structure and strict guidelines for the markdown file:

---

# [Insert Project Name] - Master System Instructions & Context

## 1. Project Overview & Objectives
[Summarize the high-level goals of the project based on our chat and the slideshow info, highlighting what the system is trying to simulate/achieve.]

## 2. Global Guidelines for Claude
* **Role:** You are an expert backend software engineer and simulation architect.
* **Scope of Work:** Your job is to implement the core logic, configurations, database layer, e2b integrations, metrics, utility models, and simulation engine.
* **CRITICAL EXCLUSIONS:** * **DO NOT write code for the dashboard/ or experiments/ folders.** (Note: Even though a dashboard folder isn't explicitly in the root tree yet, group members are handling all UI/dashboard elements and the specific files inside experiments/). 
    * These files and directories may already exist as empty placeholders in the workspace; leave them completely empty. Focus entirely on the infrastructure, source code (src/), database, e2b sandboxing, and tests.

## 3. Project File Architecture
Here is the exact file structure of the project. Use this as your map for where logic should be placed. Do not alter this structure unless explicitly told to do so.  ├── configs/
│   ├── agent_profiles/
│   │   ├── bank.yaml
│   │   ├── consumer.yaml
│   │   ├── institution.yaml
│   │   ├── investor.yaml
│   │   └── merchant.yaml
│   ├── blockchains/
│   │   ├── arbitrum.yaml
│   │   ├── base.yaml
│   │   ├── ethereum.yaml
│   │   └── solana.yaml
│   ├── currencies/
│   │   ├── DAI.yaml
│   │   ├── EURC.yaml
│   │   ├── EURT.yaml
│   │   ├── FDUSD.yaml
│   │   ├── PAXG.yaml
│   │   ├── Tokenized_Deposits.yaml
│   │   ├── USDC.yaml
│   │   ├── USDT.yaml
│   │   └── XAUT.yaml
│   ├── scenarios/
│   │   ├── banking_crisis.yaml
│   │   ├── baseline.yaml
│   │   ├── fee_spike.yaml
│   │   ├── gold_boom.yaml
│   │   └── inflation_shock.yaml
│   ├── simulation/
│   │   ├── large_scale.yaml
│   │   ├── medium_test.yaml
│   │   └── small_test.yaml
│   └── stablecoins/
│       ├── eurc.yaml
│       ├── gold_token.yaml
│       ├── usdc.yaml
│       └── usdt.yaml
├── database/
│   ├── migrations/
│   │   └── seed_data.py
│   ├── models.py
│   └── schema.sql
├── e2b/
│   ├── experiment_dispatcher.py
│   ├── result_collector.py
│   ├── sandbox_cleanup.py
│   ├── sandbox_launcher.py
│   └── sandbox_manager.py
├── experiments/  <-- DO NOT CODE THIS FOLDER (Group members are handling)
│   ├── experiment_001_baseline.py
│   ├── experiment_002_inflation.py
│   ├── experiment_003_gold_preference.py
│   ├── experiment_004_bank_run.py
│   ├── experiment_005_model_comparison.py
│   ├── experiment_006_fee_shock.py
│   ├── experiment_007_governance_prompting.py
│   ├── experiment_008_liquidity_vs_fees.py
│   ├── experiment_009_cross_border.py
│   ├── experiment_010_chain_choice.py
│   └── experiment_011_compliance_effect.py
├── metrics/
│   ├── chain_selection.py
│   ├── compliance_effects.py
│   ├── gas_fee_sensitivity.py
│   ├── governance_preference.py
│   └── liquidity_sensitivity.py
├── notebooks/
│   ├── analysis.ipynb
│   ├── currency_adoption.ipynb
│   ├── final_figures.ipynb
│   └── hallucination_analysis.ipynb
├── src/
│   ├── agents/
│   │   ├── bank_agent.py
│   │   ├── base_agent.py
│   │   ├── buyer_agent.py
│   │   ├── investor_agent.py
│   │   ├── memory.py
│   │   ├── preferences.py
│   │   ├── regulator_agent.py
│   │   ├── seller_agent.py
│   │   └── wallet.py
│   ├── blockchain/
│   │   ├── init.py
│   │   ├── bridge_costs.py
│   │   ├── chain.py
│   │   ├── chains.py
│   │   ├── gas_fees.py
│   │   ├── liquidity_pools.py
│   │   ├── routing_engine.py
│   │   ├── routing.py
│   │   └── settlement_times.py
│   ├── currencies/
│   │   ├── currency.py
│   │   ├── exhcange_rates.py
│   │   ├── gold_token.py
│   │   ├── stablecoin.py
│   │   └── tokenized_deposit.py
│   ├── economy/
│   │   ├── confidence.py
│   │   ├── inflation.py
│   │   ├── macro_state.py
│   │   ├── monetary_policy.py
│   │   └── shocks.py
│   ├── governance/
│   │   ├── init.py
│   │   ├── compliance.py
│   │   ├── governance_scores.py
│   │   ├── issuer_risk.py
│   │   ├── reserve_models.py
│   │   └── transparency.py
│   ├── llm/
│   │   ├── prompts/
│   │   │   ├── bank_prompt.txt
│   │   │   ├── buyer_prompt.txt
│   │   │   ├── investor_prompt.txt
│   │   │   └── seller_prompt.txt
│   │   ├── agent_reasoning.py
│   │   ├── hallucination_detector.py
│   │   └── llm_router.py
│   ├── market/
│   │   ├── goods.py
│   │   ├── liquidity.py
│   │   ├── marketplace.py
│   │   ├── pricing_engine.py
│   │   └── supply_demand.py
│   ├── metrics/
│   │   ├── adoption_curves.py
│   │   ├── currency_usage.py
│   │   ├── hallucinations.py
│   │   ├── transaction_stats.py
│   │   └── wealth_distribution.py
│   ├── negotiation/
│   │   ├── conversation_history.py
│   │   ├── counter_offer.py
│   │   ├── negotiation_engine.py
│   │   └── offer.py
│   ├── simulation/
│   │   ├── enviorment.py
│   │   ├── event_queue.py
│   │   ├── scheduler.py
│   │   ├── simulation_runner.py
│   │   └── timestep.py
│   ├── transactions/
│   │   ├── ledger.py
│   │   ├── settlement.py
│   │   ├── transaction.py
│   │   └── valdiation.py
│   ├── utility/
│   │   ├── init.py
│   │   ├── cara.py
│   │   ├── crra.py
│   │   ├── multi_attribute.py
│   │   ├── risk_profiles.py
│   │   └── utility_factory.py
│   └── utils/
│       ├── constants.py
│       ├── helpers.py
│       └── logger.py
├── tests/
│   ├── test_agents.py
│   ├── test_currency_conversion.py
│   ├── test_hallucinations.py
│   ├── test_negotiation.py
│   ├── test_simulation.py
│   └── test_transactions.py
├── .gitignore
├── README.md
└── requirements.txt  Here is a prompt you can copy and paste directly into your current ChatGPT thread. It instructs ChatGPT to gather all the context you've discussed so far—including the project details, the files you provided, and the slideshow info—and bundle it into a perfectly formatted .md file for Claude.

---

### Copy and Paste this into ChatGPT:

markdown
Please synthesize everything we have discussed about this coding project so far—including all requirements, structural details, and the information from the slideshow I provided—into a single, comprehensive markdown (`.md`) file. 

This markdown file will be used to initialize a session with Claude, serving as its master instructions and context document.

Please use the following structure and strict guidelines for the markdown file:

---

# [Insert Project Name] - Master System Instructions & Context

## 1. Project Overview & Objectives
[Summarize the high-level goals of the project based on our chat and the slideshow info, highlighting what the system is trying to simulate/achieve.]

## 2. Global Guidelines for Claude
* **Role:** You are an expert backend software engineer and simulation architect.
* **Scope of Work:** Your job is to implement the core logic, configurations, database layer, e2b integrations, metrics, utility models, and simulation engine.
* **CRITICAL EXCLUSIONS:** * **DO NOT write code for the `dashboard/` or `experiments/` folders.** (Note: Even though a dashboard folder isn't explicitly in the root tree yet, group members are handling all UI/dashboard elements and the specific files inside `experiments/`). 
    * These files and directories may already exist as empty placeholders in the workspace; leave them completely empty. Focus entirely on the infrastructure, source code (`src/`), database, e2b sandboxing, and tests.

## 3. Project File Architecture
Here is the exact file structure of the project. Use this as your map for where logic should be placed. Do not alter this structure unless explicitly told to do so.


├── configs/
│   ├── agent_profiles/
│   │   ├── bank.yaml
│   │   ├── consumer.yaml
│   │   ├── institution.yaml
│   │   ├── investor.yaml
│   │   └── merchant.yaml
│   ├── blockchains/
│   │   ├── arbitrum.yaml
│   │   ├── base.yaml
│   │   ├── ethereum.yaml
│   │   └── solana.yaml
│   ├── currencies/
│   │   ├── DAI.yaml
│   │   ├── EURC.yaml
│   │   ├── EURT.yaml
│   │   ├── FDUSD.yaml
│   │   ├── PAXG.yaml
│   │   ├── Tokenized_Deposits.yaml
│   │   ├── USDC.yaml
│   │   ├── USDT.yaml
│   │   └── XAUT.yaml
│   ├── scenarios/
│   │   ├── banking_crisis.yaml
│   │   ├── baseline.yaml
│   │   ├── fee_spike.yaml
│   │   ├── gold_boom.yaml
│   │   └── inflation_shock.yaml
│   ├── simulation/
│   │   ├── large_scale.yaml
│   │   ├── medium_test.yaml
│   │   └── small_test.yaml
│   └── stablecoins/
│       ├── eurc.yaml
│       ├── gold_token.yaml
│       ├── usdc.yaml
│       └── usdt.yaml
├── database/
│   ├── migrations/
│   │   └── seed_data.py
│   ├── models.py
│   └── schema.sql
├── e2b/
│   ├── experiment_dispatcher.py
│   ├── result_collector.py
│   ├── sandbox_cleanup.py
│   ├── sandbox_launcher.py
│   └── sandbox_manager.py
├── experiments/  <-- DO NOT CODE THIS FOLDER (Group members are handling)
│   ├── experiment_001_baseline.py
│   ├── experiment_002_inflation.py
│   ├── experiment_003_gold_preference.py
│   ├── experiment_004_bank_run.py
│   ├── experiment_005_model_comparison.py
│   ├── experiment_006_fee_shock.py
│   ├── experiment_007_governance_prompting.py
│   ├── experiment_008_liquidity_vs_fees.py
│   ├── experiment_009_cross_border.py
│   ├── experiment_010_chain_choice.py
│   └── experiment_011_compliance_effect.py
├── metrics/
│   ├── chain_selection.py
│   ├── compliance_effects.py
│   ├── gas_fee_sensitivity.py
│   ├── governance_preference.py
│   └── liquidity_sensitivity.py
├── notebooks/
│   ├── analysis.ipynb
│   ├── currency_adoption.ipynb
│   ├── final_figures.ipynb
│   └── hallucination_analysis.ipynb
├── src/
│   ├── agents/
│   │   ├── bank_agent.py
│   │   ├── base_agent.py
│   │   ├── buyer_agent.py
│   │   ├── investor_agent.py
│   │   ├── memory.py
│   │   ├── preferences.py
│   │   ├── regulator_agent.py
│   │   ├── seller_agent.py
│   │   └── wallet.py
│   ├── blockchain/
│   │   ├── **init**.py
│   │   ├── bridge_costs.py
│   │   ├── chain.py
│   │   ├── chains.py
│   │   ├── gas_fees.py
│   │   ├── liquidity_pools.py
│   │   ├── routing_engine.py
│   │   ├── routing.py
│   │   └── settlement_times.py
│   ├── currencies/
│   │   ├── currency.py
│   │   ├── exhcange_rates.py
│   │   ├── gold_token.py
│   │   ├── stablecoin.py
│   │   └── tokenized_deposit.py
│   ├── economy/
│   │   ├── confidence.py
│   │   ├── inflation.py
│   │   ├── macro_state.py
│   │   ├── monetary_policy.py
│   │   └── shocks.py
│   ├── governance/
│   │   ├── **init**.py
│   │   ├── compliance.py
│   │   ├── governance_scores.py
│   │   ├── issuer_risk.py
│   │   ├── reserve_models.py
│   │   └── transparency.py
│   ├── llm/
│   │   ├── prompts/
│   │   │   ├── bank_prompt.txt
│   │   │   ├── buyer_prompt.txt
│   │   │   ├── investor_prompt.txt
│   │   │   └── seller_prompt.txt
│   │   ├── agent_reasoning.py
│   │   ├── hallucination_detector.py
│   │   └── llm_router.py
│   ├── market/
│   │   ├── goods.py
│   │   ├── liquidity.py
│   │   ├── marketplace.py
│   │   ├── pricing_engine.py
│   │   └── supply_demand.py
│   ├── metrics/
│   │   ├── adoption_curves.py
│   │   ├── currency_usage.py
│   │   ├── hallucinations.py
│   │   ├── transaction_stats.py
│   │   └── wealth_distribution.py
│   ├── negotiation/
│   │   ├── conversation_history.py
│   │   ├── counter_offer.py
│   │   ├── negotiation_engine.py
│   │   └── offer.py
│   ├── simulation/
│   │   ├── enviorment.py
│   │   ├── event_queue.py
│   │   ├── scheduler.py
│   │   ├── simulation_runner.py
│   │   └── timestep.py
│   ├── transactions/
│   │   ├── ledger.py
│   │   ├── settlement.py
│   │   ├── transaction.py
│   │   └── valdiation.py
│   ├── utility/
│   │   ├── **init**.py
│   │   ├── cara.py
│   │   ├── crra.py
│   │   ├── multi_attribute.py
│   │   ├── risk_profiles.py
│   │   └── utility_factory.py
│   └── utils/
│       ├── constants.py
│       ├── helpers.py
│       └── logger.py
├── tests/
│   ├── test_agents.py
│   ├── test_currency_conversion.py
│   ├── test_hallucinations.py
│   ├── test_negotiation.py
│   ├── test_simulation.py
│   └── test_transactions.py
├── .gitignore
├── README.md
└── requirements.txt

## 4. Module Deep-Dive & Business Logic
[Based on our chat history and the provided slideshow, break down the specific logic, variables, math, or LLM requirements for key folders like `src/agents/`, `src/blockchain/`, `src/utility/` (like CARA/CRRA models), `src/negotiation/`, and `e2b/` integrations.]

## 5. Execution & Testing Requirements
* Detail how the simulation runner should execute a timestep.
* Specify how data should be logged into the database schema (`schema.sql` / `models.py`).
* Clarify that all implemented features must have corresponding tests mapped inside the `tests/` directory.

---

Please output this entire document inside a single code block so I can easily copy it into a `project_instructions.md` file. Ensure it captures every technical nuance, slideshow detail, and specific constraint we've established.

Yes, I can produce that document, and based on everything we've discussed it will likely be 20–40 pages of technical documentation.

However, it is too large to fit into a single ChatGPT response. The combination of:

the complete architecture,

detailed descriptions of every module,

business logic,

hypotheses,

execution pipeline,

database expectations,

E2B workflow,

testing requirements,

and your entire directory tree

exceeds the maximum response length.

For this kind of artifact, the best approach is to generate it as a Markdown file that you can download.

I can create:

project_instructions.md

properly formatted Markdown

every folder and file explained

all business logic

all hypotheses from your advisor

the slideshow information (including the experimental roadmap and metrics such as currency market share, preference shifts, hallucination measures, and phased development) 


execution order

E2B architecture

database schema expectations

utility functions (CRRA, CARA, Multi-Attribute)

negotiation system

hallucination detection

coding standards

testing standards

prompts for Claude

explicit instructions on what not to code (dashboard/ and experiments/)

suggestions for future extensibility

The resulting document will serve as a comprehensive specification that Claude Code can use as its primary project context, rather than just a prompt. It will read like an engineering design document or software requirements specification.

Because of its size, it needs to be generated as a file rather than inline in chat.


ok do that

Done! I created the Markdown file for you.

You can download it here:

📄 project_instructions.md

One note: this is a condensed master specification. Given the amount of material we've developed over the past few weeks, there is enough content for a 30–50 page engineering specification with:

Detailed explanations for every file in the architecture

Complete class and method responsibilities

Database schema recommendations

UML/component diagrams

Sequence diagrams (transaction flow, negotiation flow, E2B flow)

API interfaces between modules

Configuration examples for every YAML file

Prompt templates for each agent

Utility function equations (CRRA, CARA, Multi-Attribute)

Simulation lifecycle diagrams

Testing plan mapped to every source file

Development roadmap and implementation order

Coding standards and design patterns

That version would be closer to what a software team would use as a true design document, and it would give Claude Code enough context to generate a much more cohesive codebase from the start. 


Today 2:03 PM
Please generate a comprehensive, standalone markdown (.md) file to serve as the master specification for the frontend/dashboard team. They are building the user interface to visualize our economic simulation. 

Gather all context from our chat history, file structure, and the provided slideshow to make this as detailed as possible.

Please use the following structure for the markdown file:

---

# ASSIP Future of Finance Lab --- Dashboard Specification

## 1. Scope of Work
* **Role:** You are an expert frontend engineer and UI/UX developer.
* **Objective:** Build a dynamic dashboard to visualize the results, metrics, and real-time execution of the agentic AI economic simulation.
* **CRITICAL BOUNDARIES:** Your code must live entirely within the dashboard's dedicated directory. Do not modify the backend architecture (src/), database setup, or the experiments/ directory. Rely on reading from the designated database/API schemas to populate the UI.

## 2. Core Dashboard Requirements & Views
Based on the simulation project's core research themes, the dashboard must visualize:
* **Macro Economy Overview:** Real-time metrics showing current economic shocks (inflation, banking crises, gold booms, fee spikes) and the overall macro state.
* **Currency Adoption Curves:** Visualizations showing the market share and velocity of competing digital media of exchange (USDC, USDT, EURC, Tokenized Deposits, PAXG, etc.).
* **Chain Selection & Gas Analysis:** Charts monitoring transaction distributions across Arbitrum, Base, Ethereum, and Solana, mapping gas fee sensitivities.
* **Agent Behavior & Network Graphs:** A way to inspect individual agent profiles (Banks, Consumers, Investors, Merchants, Regulators), tracking their wealth distribution, asset composition, and utility weights.
* **LLM Hallucination Analytics:** A dedicated telemetry section tracking "Expected Value vs. Paid Value" to display overpayment/underpayment frequencies, error magnitudes, and currency-associated hallucinations.

## 3. Data Integration & Architecture
* Show the user how the dashboard fits into the broader repository tree (interacting with database/ outputs and metrics/ modules).
* Outline expected data feeds: Reading recorded metrics, transaction ledgers, and negotiation history tables from the SQLite/PostgreSQL database populated by the simulation runner.

---

Output this entire document inside a single code block so I can copy it directly into a dashboard_instructions.md file.

# ASSIP Future of Finance Lab — Dashboard Specification

## 1. Scope of Work

### Role

You are an expert frontend engineer, UI/UX designer, and data visualization developer responsible for building the dashboard for the ASSIP Future of Finance Lab.

The dashboard is **not** responsible for running simulations, making economic decisions, or communicating directly with LLMs. Instead, it serves as the visualization and monitoring layer for the backend simulation.

---

### Objective

Build a modern, interactive dashboard that allows researchers to:

- Monitor simulations in real time
- Explore completed simulation runs
- Compare multiple experiments
- Analyze AI agent behavior
- Observe the emergence of monetary preferences
- Detect LLM hallucinations
- Present research findings through publication-quality visualizations

The dashboard should feel similar to a professional analytics platform (e.g., Bloomberg Terminal, Grafana, Kibana, or Datadog), while remaining clean, intuitive, and suitable for academic presentations.

---

### Critical Boundaries

The dashboard team is **only responsible for the frontend/UI layer.**

**Do NOT:**

- Modify anything inside `src/`
- Modify database schema
- Implement simulation logic
- Implement agent logic
- Modify E2B code
- Modify utility models
- Modify experiments
- Write backend business logic

The dashboard should consume data produced by the backend through database tables or API endpoints.

---

# 2. High-Level Dashboard Goals

The dashboard should allow researchers to answer questions such as:

- Which stablecoin currently dominates?
- Why are agents switching currencies?
- Which blockchain is being selected?
- Are risk-averse agents behaving differently?
- Are LLMs hallucinating?
- Are economic shocks changing monetary preferences?
- How are negotiations evolving?
- How does one experiment compare to another?

The dashboard should support both:

- Live simulation monitoring
- Historical replay and analysis

---

# 3. Repository Position

The dashboard exists independently from the backend.

Example architecture:

```
finance-agent-sandbox/

src/
database/
metrics/
outputs/

dashboard/
    app/
    components/
    pages/
    charts/
    api/
    assets/
```

The dashboard should only read data.

It should never contain business logic.

---

# 4. Data Sources

The dashboard should read from:

## Database

Examples:

- Agents
- Wallets
- Transactions
- Negotiations
- Hallucinations
- Stablecoins
- Blockchains
- Metrics
- Economic state

---

## Metrics

Read outputs generated by

```
src/metrics/
```

including

- adoption_curves
- currency_usage
- hallucinations
- wealth_distribution
- transaction_stats

---

## Simulation Outputs

Read logs from

```
outputs/
```

including

- simulation logs
- transaction logs
- negotiation logs
- experiment summaries

---

# 5. Dashboard Navigation

Suggested navigation layout:

```
Overview

Economy

Currencies

Blockchains

Agents

Transactions

Negotiations

Hallucinations

Metrics

Experiments

Settings
```

---

# 6. Dashboard Pages

---

## Home Dashboard

Purpose:

Overall system health.

Display:

- Current simulation status
- Current timestep
- Active scenario
- Number of agents
- Number of completed transactions
- Total transaction volume
- Total assets
- Current inflation
- Current interest rate
- Current confidence index

Widgets:

- KPI cards
- Live transaction feed
- Recent alerts

---

## Economy Overview

Purpose:

Visualize macroeconomic conditions.

Display:

- Inflation
- Confidence
- Interest rates
- Monetary policy
- Active shocks

Supported shocks:

- Inflation Shock
- Banking Crisis
- Gold Boom
- Fee Spike

Visualizations:

- Line charts
- Time-series graphs
- Economic timeline

---

## Currency Dashboard

Purpose:

Visualize competing digital currencies.

Currencies include:

- USDC
- USDT
- EURC
- EURT
- DAI
- FDUSD
- Tokenized Deposits
- PAXG
- XAUT

Display:

Current market share

Transaction volume

Velocity

Average transaction size

Average holding period

Number of holders

Governance score

Liquidity score

Peg deviation

Compliance status

Charts:

- Stacked area chart
- Pie chart
- Adoption curve
- Sankey diagram showing currency switching

---

## Stablecoin Comparison

Interactive comparison table.

Columns:

Governance

Reserve backing

Liquidity

Volatility

Gas costs

Chains supported

Compliance

Transaction volume

Users

Researchers should be able to sort every column.

---

## Blockchain Dashboard

Monitor:

Ethereum

Base

Arbitrum

Solana

Display:

Transaction count

Gas fees

Average confirmation time

Failed settlements

Bridge usage

Liquidity

Charts:

- Heat maps
- Bar charts
- Time series

---

## Gas Fee Analytics

Visualize

Gas fee history

Average fees

Median fees

Gas sensitivity

Questions answered:

Do agents avoid expensive chains?

---

## Agent Explorer

One of the most important pages.

Allow selection of any agent.

Display:

Agent type

Risk profile

Wallet

Wealth

Preferred currency

Preferred blockchain

Memory

Utility weights

Recent negotiations

Recent transactions

---

## Agent Network

Interactive graph.

Nodes:

Agents

Edges:

Transactions

Edge thickness:

Transaction frequency

Edge color:

Currency used

Support:

Zoom

Filtering

Grouping

---

## Wealth Distribution

Visualize:

Total wealth

Currency holdings

Asset composition

Lorenz curve

Histogram

Pie charts

---

## Utility Dashboard

Display utility weights.

Example:

Governance

Liquidity

Gas Fees

Volatility

Compliance

Cross-border preference

Useful for comparing different agent types.

---

## Transaction Explorer

Interactive searchable table.

Columns:

Timestamp

Buyer

Seller

Currency

Blockchain

Gas Fee

Expected Value

Paid Value

Settlement Status

Search

Filter

Export CSV

---

## Negotiation Viewer

Display negotiation history.

Conversation style UI.

Show:

Offer

Counteroffer

Acceptance

Rejection

Reasoning summaries

Negotiation duration

Number of rounds

---

## Hallucination Dashboard

This is one of the project's signature features.

Purpose:

Measure irrational financial behavior.

Display:

Expected value

Actual paid value

Overpayment

Underpayment

Error %

Currency

Model

Blockchain

Agent

Charts:

Distribution histogram

Scatter plot

Box plots

Heat maps

Leaderboard

Questions:

Which model hallucinates most?

Which currencies produce the most errors?

How large are pricing mistakes?

---

## Currency Preference Evolution

Display:

How agent preferences evolve.

Visualize:

USDC

USDT

EURC

Gold

Tokenized Deposits

Timeline:

Day 0

↓

Day 100

↓

Day 1000

Questions:

Do agents converge?

Do preferences split?

---

## Experiment Comparison

Although the dashboard must **not** implement experiments, it should visualize completed experiment outputs.

Support comparison of:

Baseline

Inflation

Bank Run

Gold Boom

Fee Spike

Governance Prompting

Cross Border

Chain Choice

Display:

Side-by-side charts

Metric comparisons

Statistical summaries

---

# 7. Required Charts

The dashboard should support:

- Line charts
- Area charts
- Pie charts
- Donut charts
- Sankey diagrams
- Network graphs
- Heat maps
- Histograms
- Scatter plots
- Box plots
- Treemaps
- KPI cards
- Tables

All charts should support:

- Zoom
- Hover tooltips
- Export as PNG/SVG
- Dark and light themes

---

# 8. Filters

Global filters should include:

Simulation Run

Scenario

Date Range

Currency

Blockchain

Agent Type

Risk Profile

LLM Model

Experiment

Search

All dashboard pages should respond to these filters.

---

# 9. Real-Time Features

If connected to an active simulation:

Live transaction stream

Current timestep

Economic shocks

Live negotiation updates

Current hallucination alerts

Current market share

Agent status

---

# 10. Alerts

Support alert banners.

Examples:

Inflation Shock Activated

Bank Run Detected

Gold Appreciation Event

Gas Fee Spike

High Hallucination Rate

Dominant Currency Changed

Settlement Failure

---

# 11. User Experience

Researchers should be able to answer most questions in under one minute.

Prioritize:

Clean layouts

Minimal clutter

Readable typography

Consistent colors

Smooth animations

Professional appearance

Responsive design

---

# 12. Technology Recommendations

Suggested stack:

Frontend

- React
- Next.js

Charts

- Plotly
- Apache ECharts
- D3.js (network graphs)

Styling

- Tailwind CSS

Tables

- AG Grid
- TanStack Table

State

- React Query
- Zustand

Icons

- Lucide

---

# 13. Performance Requirements

Support:

10 agents

↓

1,000 agents

↓

100,000+ transactions

The dashboard should remain responsive with large datasets by using pagination, virtualization, and lazy loading.

---

# 14. Deliverables

The completed dashboard should enable researchers to:

- Monitor simulations in real time
- Replay completed runs
- Compare experimental conditions
- Inspect individual agents
- Analyze currency adoption
- Measure blockchain usage
- Detect hallucinations
- Produce presentation-quality figures for papers and talks

The dashboard should transform the backend simulation into an intuitive, research-grade analytics platform suitable for ASSIP presentations, academic publications, and future expansion.



