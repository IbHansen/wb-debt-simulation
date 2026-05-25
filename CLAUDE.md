# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **debt simulation and analytical toolkit** for modeling sovereign debt portfolios. The project uses **ModelFlow** (via ModelFlowIb package), a domain-specific language framework for building scalable economic/financial models using **lists** and **templates** that expand automatically across multiple entities.

The core concept: instead of writing separate equations for each bond/maturity/currency combination, you define a single templated equation and ModelFlow expands it across all relevant list members dynamically.

### Repository Structure

`
wb-debt-simulation/
├── simulation/              # Original simulation models (v1)
│   ├── model/              # Compiled ModelFlow models (.pcim files)
│   ├── graph/              # Generated model dependency graphs
│   ├── Specification_debt_simulation_model.ipynb    # Bond portfolio simulation spec
│   ├── debt_issurance_and_market_scenarier.ipynb   # Market scenario testing
│   └── simulations.py      # UI functions for interactive simulation widgets
│
├── simulation_2/            # Current/updated simulation models (v2)
│   ├── model/
│   │   ├── port.pcim       # Portfolio model (main model for debt scenarios)
│   │   └── hist.pcim       # Historical debt model (for backtesting)
│   ├── graph/              # Model graphs/visualizations
│   ├── Specification_debt_simulation_model.ipynb     # Portfolio spec (updated)
│   ├── Specification_debt_hist_model.ipynb          # Historical spec
│   ├── debt_issurance_and_market_scenarier.ipynb   # Scenario analysis
│   └── simulations.py      # UI widget code
│
├── optimization/           # Currency/FX optimization analysis
│   ├── exchangerates_get.py     # ECB FX rate data fetching
│   └── currency.ipynb           # Currency frontier optimization
│
├── test_interactive/       # Interactive testing & experimentation
│   ├── interactive.ipynb
│   ├── interactive-clause.ipynb
│   └── test_*.ipynb
│
└── Material/              # Reference documents
    └── stress-testing-imf-sarb.pdf
`

## ModelFlow Skill Files

Detailed ModelFlow reference docs are in `C:\modelflow-skill\references\`:

- **`model-construction.md`** — FRML syntax, equation flags (`<DAMP>`, `<STOC>`, `<EXO>`), normalization, `.equpdate`/`.eqdelete`, `Mexplode` DSL, saving/loading `.pcim` files.
- **`makemodel.md`** — `Makemodel` class: markdown `>` equation format, tags (`<ident>`, `<estimator=...>`, `<stoc>`), LIST/TLIST/DOABLE templating, `replacements=`, estimation integration, `init_addfactors`.

## Key Concepts

### ModelFlow Framework

ModelFlow is a declarative economic/financial modeling framework. Key features:

- **Lists and Sublists**: Define groups of entities with attributes
  - Example: A bond list with sublists for maturity, grace period, currency, domestic/foreign flag
  - Expansion: issued_2025 * issued_2050 automatically generates issued_2025, issued_2026, ..., issued_2050
  
- **Variable Naming**: `__` (double underscore) is the dimension separator. Each dimension of a variable is joined with `__`.
  - Pattern: `{concept}__{dim1}__{dim2}` — e.g. `stock_ultimo__1_year_dom__issued_2026`
  - Aggregates use `__all` — e.g. `interest_payments__all`
  - Currency aggregates — e.g. `stock_ultimo__dom`, `stock_ultimo__usd`

- **Variable Expansion**: Template equations expand across list members
  - Write once: `amortizing__bond = outstanding_in_currency__bond * amortizing_share`
  - Generates: `amortizing__1_year_dom`, `amortizing__5_year_dom`, ..., `amortizing__10_year_usd`, etc.

- **Logical Variables**: 0/1 binary variables for time-dependent logic
  - Example: logical_issuing__bond = (current_year == issued_year) triggers only in issuance year
  - Example: logical_amortizing__bond = (issued_year + grace) < current_year <= (issued_year + maturity)

- **Model Segments**: Models are organized into named sections (e.g., "logical", "funding", "issurance", "stock", "interest")

- **Model Files**: Compiled models are .pcim files loaded via model.modelload('model/filename')

### Portfolio Model (port.pcim)

The main simulation model tracks sovereign debt portfolio dynamics:

**Lists**:
- bond: 6 bond types (1/5/10-year domestic and USD foreign) with maturity, grace, currency attributes
- issued: issuance years (2025-2050)
- currency: domestic (dom) and foreign (usd)

**Key Variables**:
- new_issue__bond: Amount issued of each bond type
- stock_primo/ultimo__bond: Beginning/end-of-year debt stock (in currency)
- amortizing__bond: Principal repayment schedule
- interest_payments__bond: Interest costs by bond
- funding_need: Deficit + amortization + interest = total borrowing need

**Exogenous Inputs**:
- DEFICIT: Annual government deficit
- BOND_SHARE_DOM: % of funding from domestic vs. foreign markets
- NEW_ISSUE_SHARE__bond: Policy choice on which bonds to issue
- INTEREST_RATE__currency__maturity: Market interest rates
- DOM_USD, DOM_DOM: FX rates (currency conversion)

**Dynamics**:
1. Calculate funding need (deficit + amortization + interest)
2. Split between domestic/foreign based on policy
3. Allocate to specific bonds based on issuance shares
4. Track outstanding stock and interest accrual by bond
5. Convert between currencies using FX rates

### Historical Model (hist.pcim)

A simplified model for backward-looking analysis. Uses:
- Fixed bond list (e.g., 10-year domestic and USD bonds with specific balances)
- Fixed issuance year (2025)
- Tracks amortization and interest payments over time
- No endogenous policy decisions (bonds/rates are fixed inputs)

## Common Workflows

### Running Simulations in Jupyter

1. **Load a model**:
   from modelclass import model
   mport, baseline = model.modelload('model/port', run=1)
   Returns model object and baseline DataFrame of results

2. **Access variables**:
   mport['stock_ultimo__all INTEREST_PAYMENTS__ALL'].df  # Get results
   mport.NEW_ISSUE__1_YEAR_DOM  # Inspect a single variable definition

3. **Create input DataFrame and run**:
   import pandas as pd
   df = pd.DataFrame([...], index=years, columns=['CURRENT_YEAR', ...])
   res = mport(df, start_year, end_year, silent=0)  # silent=0 shows solve progress

4. **Interactive UI** (in Jupyter):
   from simulations import ui_simulations
   ui_simulations()  # Creates tabbed widget for policy experimentation

### Modifying Models

Model definitions are in .ipynb notebooks using the %%Mexplodemodel magic command:

%%Mexplodemodel port segment=issurance replacements=... nshow render=0
## Model equations in natural form
> new_issue__bond = new_issue_share__{bond}/100 * issued_domestic
> amortizing__bond = outstanding_in_currency__bond * amortizing_share

Key points:
- Curly braces {variable} reference sublist values
- Angle brackets <sum=all> aggregate across sublist members
- doable keyword marks logical (0/1) variables
- Replacements allow templating across different model configurations

After editing, notebooks export compiled models to .pcim files via mport.modeldump('model/port').

### Generating Model Graphs

mport.drawmodel(svg=True, browser=True)  # Visualize variable dependencies

Output: Graphviz .gv, .svg, .png files showing formula dependencies.

## Dependencies

Main Python packages (from notebooks):
- **ModelFlowIb**: The core modeling framework
- **pandas**: Data manipulation and results handling
- **numpy**: Numerical computation
- **requests**: HTTP (for FX rate fetching)
- **cvxopt**: Optimization (currency frontier analysis)
- **IPython.display**: Jupyter output rendering

In Google Colab, dependencies auto-install via:
os.system('pip -qqq install ModelFlowIb')

## Tips for Code Changes

- **Model logic changes**: Edit the .ipynb notebooks, re-run cells to recompile, then dump new .pcim files
- **Widget/UI changes**: Edit simulations.py functions directly
- **FX data**: exchangerates_get.py provides utilities to fetch ECB rates programmatically
- **Testing scenarios**: Use test_interactive/ notebooks to experiment with new assumptions
- **Version control**: .pcim model files are binary; track the source .ipynb notebooks in git

## Running in Google Colab

Notebooks include Colab badges at the top. When opened in Colab:
- Dependency installation happens automatically
- GraphViz requires: apt install graphviz
- Results display in-notebook
- Share/collaborate via Colab link

## Important Files

| File | Purpose |
|------|---------|
| simulation_2/Specification_debt_simulation_model.ipynb | Main portfolio model spec (defines bonds, lists, equations) |
| simulation_2/Specification_debt_hist_model.ipynb | Historical backtest model spec |
| simulation_2/simulations.py | Interactive UI widget definitions |
| simulation_2/model/port.pcim | Compiled portfolio model (binary) |
| simulation_2/model/hist.pcim | Compiled historical model (binary) |
| optimization/exchangerates_get.py | FX rate data retrieval |
| optimization/currency.ipynb | Currency allocation frontier analysis |
| Material/stress-testing-imf-sarb.pdf | Reference documentation |

## Model Output Structure

Results are pandas DataFrames indexed by year with columns for each variable:
- Aggregated variables: STOCK_ULTIMO__ALL, INTEREST_PAYMENTS__ALL, etc.
- By-bond variables: STOCK_ULTIMO__1_YEAR_DOM__ISSUED_2026, INTEREST_PAYMENTS__5_YEAR_USD__ISSUED_2027, etc.
- By-currency aggregates: STOCK_ULTIMO__DOM, STOCK_ULTIMO__USD, etc.
