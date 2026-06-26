# ASSIP-agentic-currency-simulation
Multi-agent simulation for AI medium-of-exchange preferences

# Setup
**Clone the Repo:** git clone https://github.com/adityashah522/future-of-finance-agents.git

**Make your own branch:** This prevents people from accidentally overwriting each other's work.
  ```text
  git checkout -b aditya
  git branch --set-upstream-to=origin/aditya aditya
  git config --global push.default current
  
  ```
# Workflow after:
**Every time you start working:**
```text
git pull
# 1. Download your team's latest work from GitHub
git fetch origin

# 2. Merge their work into your current workspace
git merge origin/main
```

**After making changes:**
```text
git add .
git commit -m "Added stablecoin utility function"
git push
```
**Then create a Pull Request on GitHub and merge it once it's reviewed.**


# File Arcitecture: 
```text

finance-agent-sandbox/
│
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── configs/
│   ├── currencies/
│   │   ├── usd_stablecoin.yaml
│   │   ├── euro_stablecoin.yaml
│   │   ├── gold_stablecoin.yaml
│   │   └── tokenized_deposit.yaml
│   │
│   ├── scenarios/
│   │   ├── baseline.yaml
│   │   ├── inflation_shock.yaml
│   │   ├── banking_crisis.yaml
│   │   ├── gold_boom.yaml
│   │   └── fee_spike.yaml
│   │
│   ├── agent_profiles/
│   │   ├── consumer.yaml
│   │   ├── merchant.yaml
│   │   ├── bank.yaml
│   │   ├── investor.yaml
│   │   └── institution.yaml
│   │
│   └── simulation/
│       ├── small_test.yaml
│       ├── medium_test.yaml
│       └── large_scale.yaml
│
├── src/
│   │
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── buyer_agent.py
│   │   ├── seller_agent.py
│   │   ├── investor_agent.py
│   │   ├── bank_agent.py
│   │   ├── regulator_agent.py
│   │   ├── wallet.py
│   │   ├── memory.py
│   │   └── preferences.py
│   │
│   ├── currencies/
│   │   ├── currency.py
│   │   ├── stablecoin.py
│   │   ├── gold_token.py
│   │   ├── tokenized_deposit.py
│   │   └── exchange_rates.py
│   │
│   ├── market/
│   │   ├── marketplace.py
│   │   ├── goods.py
│   │   ├── pricing_engine.py
│   │   ├── supply_demand.py
│   │   └── liquidity.py
│   │
│   ├── transactions/
│   │   ├── transaction.py
│   │   ├── ledger.py
│   │   ├── settlement.py
│   │   └── validation.py
│   │
│   ├── negotiation/
│   │   ├── negotiation_engine.py
│   │   ├── offer.py
│   │   ├── counter_offer.py
│   │   └── conversation_history.py
│   │
│   ├── llm/
│   │   ├── prompts/
│   │   │   ├── buyer_prompt.txt
│   │   │   ├── seller_prompt.txt
│   │   │   ├── investor_prompt.txt
│   │   │   └── bank_prompt.txt
│   │   │
│   │   ├── llm_router.py
│   │   ├── agent_reasoning.py
│   │   └── hallucination_detector.py
│   │
│   ├── economy/
│   │   ├── inflation.py
│   │   ├── monetary_policy.py
│   │   ├── shocks.py
│   │   ├── confidence.py
│   │   └── macro_state.py
│   │
│   ├── simulation/
│   │   ├── environment.py
│   │   ├── scheduler.py
│   │   ├── simulation_runner.py
│   │   ├── timestep.py
│   │   └── event_queue.py
│   │
│   ├── metrics/
│   │   ├── currency_usage.py
│   │   ├── hallucinations.py
│   │   ├── adoption_curves.py
│   │   ├── wealth_distribution.py
│   │   └── transaction_stats.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── helpers.py
│       └── constants.py
│
├── e2b/
│   ├── sandbox_manager.py
│   ├── sandbox_launcher.py
│   ├── sandbox_cleanup.py
│   ├── experiment_dispatcher.py
│   └── result_collector.py
│
├── database/
│   ├── schema.sql
│   ├── models.py
│   ├── migrations/
│   └── seed_data.py
│
├── experiments/
│   ├── experiment_001_baseline.py
│   ├── experiment_002_inflation.py
│   ├── experiment_003_gold_preference.py
│   ├── experiment_004_bank_run.py
│   ├── experiment_005_model_comparison.py
│   └── experiment_006_fee_shock.py
│
├── outputs/
│   ├── logs/
│   ├── transactions/
│   ├── conversations/
│   ├── metrics/
│   └── reports/
│
├── notebooks/
│   ├── analysis.ipynb
│   ├── currency_adoption.ipynb
│   ├── hallucination_analysis.ipynb
│   └── final_figures.ipynb
│
├── dashboard/
│   ├── app.py
│   ├── pages/
│   ├── charts/
│   └── components/
│
└── tests/
    ├── test_agents.py
    ├── test_transactions.py
    ├── test_currency_conversion.py
    ├── test_negotiation.py
    ├── test_hallucinations.py
    └── test_simulation.py

# Execution Flow

experiment_002_inflation.py
            ↓
sandbox_launcher.py
            ↓
environment.py
            ↓
spawn agents
            ↓
run transactions
            ↓
record negotiations
            ↓
detect hallucinations
            ↓
save results
            ↓
result_collector.py
            ↓
analysis notebook
            ↓
graphs and report


# What each file does

For a project this large, the easiest mistake is creating a bunch of folders without understanding why they exist. Here's what each part of the architecture actually does.

```

# Root Directory

```text
finance-agent-sandbox/
```

This is your entire project.

---

# Project Configuration

## `README.md`

The project homepage.

Contains:

* Project description
* Setup instructions
* How to run simulations
* Team responsibilities
* Example commands

Example:

```text
pip install -r requirements.txt
python experiments/experiment_001_baseline.py
```

---

## `requirements.txt`

Lists all Python packages required.

Example:

```text
openai
e2b
pandas
numpy
sqlalchemy
fastapi
mesa
plotly
```

---

## `.gitignore`

Prevents sensitive or unnecessary files from being uploaded to GitHub.

Very important.

Example:

```text
.env
__pycache__/
outputs/
.ipynb_checkpoints/
```

---

## `.env`

This is where API keys go.

Example:

```text
OPENAI_API_KEY=sk-xxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
E2B_API_KEY=e2b_xxxxxxxx
DATABASE_URL=postgresql://...
```

This file should **never** be pushed to GitHub.

---

# Configs Folder

```text
configs/
```

Stores experiment settings.

Think of this as the "control panel."

---

## `currencies/`

Defines currency properties.

Example:

### `usd_stablecoin.yaml`

```yaml
name: USD
transaction_fee: 0.001
volatility: 0.01
acceptance: 0.95
```

---

### `gold_stablecoin.yaml`

```yaml
name: GOLD
transaction_fee: 0.003
volatility: 0.05
acceptance: 0.45
```

---

## `scenarios/`

Economic events.

Example:

### `inflation_shock.yaml`

```yaml
day: 100
usd_inflation: 0.10
```

---

## `agent_profiles/`

Defines agent personalities.

Example:

### `consumer.yaml`

```yaml
risk_tolerance: low
prefers_stability: true
```

---

# Source Code

```text
src/
```

Contains all actual code.

---

# Agents

## `base_agent.py`

The parent class.

Every agent inherits from this.

Functions:

```python
choose_currency()
buy_goods()
update_memory()
```

---

## `buyer_agent.py`

Logic specific to buyers.

Responsibilities:

* Decide what to buy
* Negotiate prices
* Select currency

---

## `seller_agent.py`

Responsible for:

* Accepting offers
* Rejecting offers
* Setting prices

---

## `wallet.py`

Tracks balances.

Example:

```python
wallet = {
    "USD":1000,
    "EURO":500,
    "GOLD":0.2
}
```

---

## `memory.py`

Stores past experiences.

Example:

```text
USD transaction success rate = 95%
Gold transaction success rate = 40%
```

---

## `preferences.py`

Stores:

```text
USD preference = 0.7
Gold preference = 0.3
```

---

# Currencies

## `currency.py`

Base class for all currencies.

Contains:

```python
fee
volatility
exchange_rate
acceptance
```

---

## `exchange_rates.py`

Handles:

```text
USD → EURO
EURO → GOLD
GOLD → USD
```

---

# Market

## `marketplace.py`

The central exchange.

Responsible for:

* Matching buyers and sellers
* Posting available goods
* Finding counterparties

Think of it as:

```text
Amazon + stock exchange
```

---

## `goods.py`

Stores all products.

Example:

```text
cloud compute
electricity
data
AI services
```

---

## `pricing_engine.py`

Determines the true price.

Example:

```text
Cloud Compute = $100
```

This file is critical because hallucination detection depends on knowing the actual value.

---

# Transactions

## `transaction.py`

Represents a single transaction.

Example:

```text
Buyer: Agent 15
Seller: Agent 4
Price: $100
Currency: USD
```

---

## `ledger.py`

Stores every transaction permanently.

Equivalent to a blockchain ledger.

---

## `settlement.py`

Actually moves money.

Example:

```text
Buyer USD:
1000 → 900

Seller USD:
500 → 600
```

---

## `validation.py`

Checks:

* sufficient funds
* accepted currency
* valid exchange rates

---

# Negotiation

## `negotiation_engine.py`

Runs conversations.

Example:

```text
Buyer offer
Seller counteroffer
Buyer accepts
```

---

## `offer.py`

Defines an offer object.

Example:

```text
100 USD for compute
```

---

## `counter_offer.py`

Handles:

```text
120 USD instead
```

---

## `conversation_history.py`

Stores negotiation logs.

---

# LLM Folder

This is the AI brain.

---

## `prompts/`

Stores prompts for agents.

Example:

### `buyer_prompt.txt`

```text
You are an AI buyer seeking to maximize purchasing power.
```

---

## `llm_router.py`

One of the most important files.

Responsible for deciding:

```text
Which model handles this request?
```

Example:

```python
buyer -> GPT
seller -> Claude
bank -> GPT
```

---

## `agent_reasoning.py`

Calls the LLM.

Example:

```python
response = client.responses.create(...)
```

This file generates decisions.

---

## `hallucination_detector.py`

Compares:

```text
True price = 100
Paid price = 300
```

Output:

```text
200% overpayment
```

---

# Economy

## `inflation.py`

Updates purchasing power.

---

## `shocks.py`

Triggers:

* inflation
* bank failures
* gold rallies

---

## `macro_state.py`

Stores the current state of the economy.

Example:

```text
inflation = 7%
interest_rate = 5%
gold_price = 3500
```

---

# Simulation

## `environment.py`

Creates the world.

---

## `scheduler.py`

Determines which agents act and when.

---

## `simulation_runner.py`

The main file.

This usually contains:

```python
for day in range(365):
    run_market()
    process_transactions()
```

---

# Metrics

## `currency_usage.py`

Tracks market share.

---

## `hallucinations.py`

Tracks pricing mistakes.

---

## `wealth_distribution.py`

Tracks inequality and concentration.

---

# E2B Folder

## `sandbox_manager.py`

Creates sandboxes.

---

## `sandbox_launcher.py`

Starts simulations inside E2B.

---

## `sandbox_cleanup.py`

Deletes finished sandboxes.

---

## `result_collector.py`

Downloads results from E2B.

---

# Database

## `models.py`

Defines database tables.

Example:

```python
Agent
Transaction
Currency
```

---

# Experiments

Each file runs a specific study.

Example:

## `experiment_001_baseline.py`

Normal economy.

---

## `experiment_002_inflation.py`

10% inflation shock.

---

## `experiment_005_model_comparison.py`

Compare GPT vs Claude vs Gemini.

---

# Outputs

Stores generated data.

```text
outputs/
    logs/
    conversations/
    metrics/
```

---

Those three folders are actually what separate a research project from just a simulation script. They serve very different purposes.

# 1. `notebooks/` — Research and Analysis

This is where you analyze results after simulations finish.

Think:

> Simulation generates data → notebooks turn data into findings.

You generally **do not put production code here**.


---

## `exploration.ipynb`

Used early in development.

Questions:

* Are transactions being recorded correctly?
* Are agents actually trading?
* Are currencies updating properly?

Example analysis:

```python
transactions.head()
transactions["currency"].value_counts()
```

---

## `currency_adoption.ipynb`

Research questions:

* Does USD dominate?
* Does gold gain market share during inflation?
* How long does it take for preferences to emerge?

Outputs:

* Market share over time
* Adoption curves
* Currency switching behavior

---

## `hallucination_analysis.ipynb`

Probably one of your most interesting notebooks.

Questions:

* How often do agents overpay?
* Which currencies produce more valuation errors?
* Which model hallucinates most frequently?

Metrics:

```text
Average overpayment %
Maximum overpayment %
Hallucination frequency
```

---

## `negotiation_analysis.ipynb`

Questions:

* How many messages are needed before agreement?
* Does longer negotiation reduce mistakes?
* Which currencies require more negotiation?

---

## `scenario_comparison.ipynb`

Compares:

| Scenario       | USD Share | Gold Share | Hallucinations |
| -------------- | --------- | ---------- | -------------- |
| Baseline       | 70%       | 10%        | 1.5%           |
| Inflation      | 45%       | 35%        | 2.1%           |
| Banking Crisis | 30%       | 20%        | 4.7%           |

This notebook may end up producing the tables for your final paper.

---

## `final_figures.ipynb`

Produces publication-quality figures:

* Currency adoption curves
* Wealth distributions
* Hallucination histograms
* Network diagrams

This notebook is usually the last step before presentations or papers.

---

# 2. `dashboard/` — Live Monitoring

The dashboard lets you watch the economy while it runs.

Think:

> notebooks = post-game analysis
> dashboard = live scoreboard

---

## Example Structure

```text
dashboard/
│
├── app.py
├── pages/
├── components/
├── charts/
└── api/
```

---

## `app.py`

The dashboard entry point.

Starts your web app.

For example:

```bash
streamlit run app.py
```

---

## `pages/`

Each page shows a different aspect of the simulation.

Example:

```text
pages/
├── market_overview.py
├── currencies.py
├── transactions.py
├── hallucinations.py
└── agents.py
```

---

## `market_overview.py`

Displays:

```text
Current simulation day
Total transactions
GDP
Inflation
```

---

## `currencies.py`

Shows:

* USD market share
* Euro market share
* Gold market share
* Exchange rates

---

## `transactions.py`

Shows live transactions:

```text
Agent 12 bought compute from Agent 4 using USD.
```

---

## `hallucinations.py`

Tracks:

```text
Agent 18 paid $400 for a $100 asset.
```

---

## `agents.py`

Displays:

* wealth rankings
* holdings
* preferences
* trust scores

---

## Why the dashboard matters

If something breaks, you'll often notice it here first.

Examples:

```text
Gold suddenly reaches 99% market share.
```

or

```text
Agent 4 paid $80 million for coffee.
```

Without a dashboard, debugging multi-agent systems becomes much harder.

---

# 3. `tests/` — Preventing Silent Failures

This folder verifies that your code behaves correctly.

For simulations this is incredibly important because bugs can look like economic behavior.

---



## `test_wallet.py`

Checks:

```python
wallet.withdraw(100)
```

Expected:

```text
1000 -> 900
```

---

## `test_transactions.py`

Verifies:

* balances update correctly
* money isn't created
* money isn't destroyed

Example:

```text
Buyer loses $100
Seller gains $100
```

---

## `test_currency_conversion.py`

Verifies:

```text
100 USD
↓
90 EUR
↓
100 USD
```

You don't want conversion bugs creating artificial arbitrage opportunities.

---

## `test_agents.py`

Checks:

* agents choose valid currencies
* agents never spend money they don't have
* preferences update properly

---

## `test_negotiation.py`

Ensures:

* conversations terminate
* infinite loops don't occur
* invalid offers are rejected

---

## `test_hallucinations.py`

Verifies:

```text
True value = 100
Paid = 150
```

Expected output:

```text
50% overpayment
```

---

## `test_simulation.py`

Runs a small economy.

Example:

```text
10 agents
100 transactions
10 days
```

Checks:

* simulation completes
* no crashes occur
* all outputs are generated

---

# Typical Workflow

```text
GitHub
   ↓
Write code
   ↓
Run tests/
   ↓
Launch simulation on E2B
   ↓
Watch dashboard/
   ↓
Store outputs/
   ↓
Analyze using notebooks/
   ↓
Generate paper figures
```

---

# Where the API keys go 

The answer is:

```text
finance-agent-sandbox/
│
├── .env
```

Example:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
E2B_API_KEY=e2b_...
DATABASE_URL=postgresql://...
```

Then load them with:

```python
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

and use them:

```python
client = OpenAI(
    api_key=OPENAI_API_KEY
)
```

---

## Important security rule

Your `.gitignore` should contain:

```text
.env
```

Otherwise someone could accidentally push API keys to GitHub.

The architecture for secrets should look like:

```text
GitHub Repository
    ↓
GitHub Secrets
    ↓
E2B Sandbox Environment Variables
    ↓
Python os.getenv()
```




