# Stablecoin and Tokenized Asset Profiles

This report analyzes each token in detail, covering their design, history, market performance, reserves, governance, use cases, and regulatory context. Where available, official documents, attestations, and reputable news are cited. Tables at the end compare tokens by peg mechanism, reserves, transparency, governance, use cases, and risks. Timeline charts (Mermaid) and price/market-cap plots illustrate key events and trends. Undefined details (e.g. CRRA/CARA usage) are noted explicitly.

## DAI (MakerDAO Stablecoin)

**Executive Summary:** DAI is a decentralized USD-pegged stablecoin managed by MakerDAO (now rebranded as Sky). It was launched in December 2017 as a crypto-collateralized stablecoin (Single-Collateral DAI or SAI) and later upgraded to Multi-Collateral DAI in Nov 2019. DAI is backed by cryptocurrency collateral locked in on-chain Maker vaults (including ETH, WBTC, LINK, stablecoins, and tokenized real-world assets like US Treasury bills). Its supply is governed by MKR token holders via DAO voting. DAI maintains a 1:1 soft peg to USD through overcollateralization and stability fees; it typically trades very close to $1, with only brief deviations during market stress (e.g. a temporary drop to ~$0.897 on Mar 11, 2023). By mid-2026, DAI circulating supply was ~4.7 billion (with Sky’s new USDS stablecoin bringing combined supply to ~$13 billion).

**Timeline of Major Events:** Key milestones include the original SAI launch (Dec 2017), the November 2019 Multi-Collateral DAI upgrade, the March 2020 “Black Thursday” Ethereum crash (DAI briefly fell below $0.90 due to liquidations), and the MakerDAO rebrand to Sky in Aug 2024 with a new USDS stablecoin. (Paxos’s Pax Gold release in 2019 and Binance listing in 2020 were notable for gold tokens in the ecosystem.) The timeline below highlights select events:

```mermaid
timeline
    title DAI (MakerDAO) Key Events
    2017-12-18 : Launch of Single-Collateral DAI (SAI) on Ethereum
    2019-11-18 : Multi-Collateral DAI (MCD) upgrade launched
    2020-03 : "Black Thursday" market crash (ETH crash; DAI briefly dipped ~$0.90)
    2022-05 : TerraUSD crash impacts stablecoin market (DAI market cap briefly fell from $8B to $6.3B)
    2024-08-27 : MakerDAO rebrands to Sky; announces USDS stablecoin (DAI upgrade opt-in) 
```

**Price & Market Cap Trends:** DAI’s price is normally ~$1, reflecting its peg. CoinMarketCap reports an all-time high of $3.67 (Nov 2019) and a low of $0.897 (Mar 2023) due to market stress. Its market capitalization peaked above $8 billion in 2022, and as of mid-2026 it is on the order of $4–5 billion. The chart below illustrates DAI’s stable price with minor deviations and its rising market cap over time (source: CoinGecko).

 *Historical DAI price (USD) and market cap (based on CoinGecko data). The peg is maintained at ~$1, with small fluctuations (e.g. SVB-2023 dip). Supply grew after 2020.*  

**Reserves & Transparency:** DAI’s “reserve” is the crypto collateral locked in Maker Vaults. All collateral holdings are on-chain, audited by anyone, and managed by smart contracts. Maker publishes audited reports and maintains transparency in Treasury assets (e.g. tokenized US Treasuries). Risk parameters (like collateralization ratios) are publicly set via governance. There is no off-chain custodian; instead, collateral can be liquidated automatically. Attestations/audits are not applicable in a traditional sense, but Maker’s contracts are audited by security firms. Maker’s treasury includes yield-bearing assets (e.g. US Treasury bills via Centrifuge) to generate revenue.

**Governance:** MakerDAO uses a decentralized governance model. MKR token holders vote on risk parameters, collateral types, and protocol changes. There is no central issuer; DAI issuance/burn is automated by smart contracts when users lock/unlock collateral. The Maker (MKR) token was rebranded to SKY (1 MKR = 24,000 SKY) when MakerDAO became Sky. Recent governance has introduced a new governance token (SKY) and a second stablecoin (USDS). All governance proposals and their outcomes are public on-chain.

**CRRA/CARA:** Not specified. MakerDAO does not use CRRA/CARA terminology; risk is managed via overcollateralization and adjustable stability fees.

**Utilities & Use Cases:** DAI is widely used in DeFi: as collateral for lending, borrowing, and earning yield; for trading on exchanges; and for payments where censorship-resistance or on-chain interoperability is needed. It also serves as on-chain liquidity for tokenized assets. MakerDAO itself uses DAI for protocol revenue, investing a portion in real-world assets to earn interest. DAI was among the first stablecoins adopted by many DeFi protocols (e.g. Compound, Aave).

**Regulatory Actions & Controversies:** MakerDAO’s decentralized nature means it has faced little direct regulatory enforcement. Notable incidents include the 2020 Black Thursday liquidation event, which caused controversy over Oracle and liquidation design. Maker introduced emergency shutdown features in response. MakerDAO underwent internal governance struggles (e.g. disputes between founders). Maker’s integration of real-world assets attracted scrutiny but is generally seen as regulatory-friendly (e.g. NYSAFE list includes MKR). No major sanctions or fines on MakerDAO have been reported. The recent rebrand and IPO of Maker to Sky (USDS/ SKY tokens) is reported and under disclosure.

## EURC (Circle Euro Coin)

**Executive Summary:** EURC is a euro-backed stablecoin issued by Circle. Launched June 30, 2022 on Ethereum, EURC is fully backed 1:1 by held euros (or Eurozone assets). It is redeemable 1:1 for EUR, with reserves held at regulated European banks and attested monthly. Circle expanded EURC to multiple chains (Avalanche, Solana, Base, Cronos, Stellar, etc) and operates under EU regulations (MiCA Electronic Money Token). It is centrally managed by Circle and its European subsidiary (Circle SAS/France). EURC is used for euro liquidity in crypto (trading, DeFi, cross-border payments) and to integrate euro into global blockchain finance. 

**Timeline:** Key dates:
- *2022-06-30*: EURC launches on Ethereum (Circle blog). Initially a Euro Coin in Centre Consortium with Circle backing.
- *2022-10*: EURC expands to Avalanche and other blockchains (avalanche announced Oct 2022 via press).
- *2023-09-26*: EURC launched on Stellar network.
- *2023-07*: MiCA regulation passes in EU (effective mid-2024), requiring stablecoins to register.
- *2024-07*: Circle obtains French EMI license, paving way under MiCA (public filings).
- *2025-11-26*: Circle publishes MiCA whitepaper for EURC compliance.
- *2026*: EU MiCA takes effect (Jan 2026); EURC is "significant" and fully compliant. 
- *2026-04*: Circle acquires Swiss banking license (or announces?), further regulatory positioning.
- *2026-07*: Circle’s MiCA notification (Nov 2025) comes into effect.

```mermaid
timeline
    title EURC (Circle Euro Coin) Events
    2022-06-30 : EURC launched on Ethereum (1:1 Euro peg)
    2022-10 : EURC added on Avalanche, Base, Cronos, Solana (multi-chain rollout)
    2023-09-26 : EURC launches on Stellar
    2024-07 : Circle obtains French EMI license (EU MiCA compliance preparation)
    2025-11-26 : Circle publishes MiCA whitepaper for EURC (EMT status)
```

**Price & Market Cap:** EURC trades at €1 (about $1.10–1.15), following EUR/USD exchange. CoinGecko market cap is ~€380M (USD$436M) as of July 2026 (supply ~380M). The chart is flat at EUR1 by design.

**Reserves & Transparency:** EURC uses a *full-reserve* model. Every EURC is backed by one euro or equivalent in high-quality euro-denominated assets. Reserves are held at regulated banks in the EU/EEA, segregated from operating funds, and attested monthly by a Big Four accounting firm. Circle publishes periodic attestations (monthly reserve reports via Deloitte) and weekly reserve breakdowns (transparency portal). Circle’s transparency page confirms USDC/EURC are fully backed by liquid reserves and “we’ve issued reports on all reserve assets since 2018”. In summary, reserve assets are 100% cash or cash-equivalent (government bills, repo) to ensure full coverage.

**Governance:** EURC is centrally issued by Circle (an SEC-registered public company) through its EU subsidiary. Governance is corporate: the Centre Consortium (Circle + Coinbase) initially overseen issuance, but in Aug 2023 Circle became sole issuer (Centre was wound down). Circle’s board and management control issuance/redemption policies. They coordinate chains and integrations. Circle also complies with regulatory governance (Board, auditors, SEC filings since IPO 2025). Users can mint/redeem via Circle accounts; there is no on-chain governance like MakerDAO.

**CRRA/CARA:** Unused/unspecified.

**Utilities & Use Cases:** EURC brings euro liquidity into crypto. It is used for euro-USD forex trades, euro-denominated payments on blockchain, DeFi lending (e.g. borrowing EURC), and cross-chain flows. Developers integrate EURC for euro transactions (like Stripe and other fintechs using USDC integrally plan to support EURC). It appeals to European entities and those seeking euro exposure without on-ramps. Being fully reserves-backed, it’s suitable for institutional treasury (e.g. corporate cash management). Circle advertises EURC for “stable global transactions” in euros.

**Regulatory & Legal:** EURC is explicitly structured as an e-money token (EMT) under EU law. Circle proactively submitted MiCA notifications and obtained EU licenses (French EMI in 2024). As of 2026, EURC is fully MiCA-compliant (registered with regulator). There have been no known enforcement actions against EURC; instead it is promoted as regulator-friendly. Potential regulatory issues include evolving EU stablecoin rules (though EURC meets them) and macro regulation of digital assets in the Eurozone. The backing at regulated banks also means standard banking risks (e.g. bank failures, though Circle’s diversity of banks mitigates this). We note **no legal controversies** specific to EURC have been reported.

## EURT (Tether Euro)

**Executive Summary:** EURT (often styled EUR₮) was Tether’s euro-backed stablecoin, pegged 1:1 to the euro. It launched on October 8, 2020 on Ethereum, issued by Tether Ltd (the same issuer as USDT). Like USDT, EURT was claimed to be fully collateralized by euro reserves, with tokens issued or redeemed by authorized parties. EURT’s usage was minor compared to USD stablecoins, and in 2025 Tether announced it would cease issuance of EURT, encouraging holders to redeem by Nov 27, 2025. The peg mechanism was simple redemption by Tether (1 EURT = €1), without on-chain collateral; it was essentially “asset-backed” by Tether’s off-chain reserves.

**Timeline:**  
```mermaid
timeline
    title EURT (Tether Euro) Events
    2020-10-08 : EURT launched on Ethereum (1 EUR peg)
    2022 : EURT issuance mostly halts (Tether stops minting new EURT)
    2025-10-27 : Tether announces wind-down of EURT issuance/redemption (deadline Nov 27, 2025)
    2025-11-27 : Official redemption deadline for EURT holders
```

- *2020-10-08*: EURT launch.  
- *2022*: New EURT issuance largely ceased (public statements).  
- *2025-10-27*: Tether announces that as of Nov 27, 2025, EURT redemptions will close. Tether cited a lack of regulatory clarity in Europe for fiat issuance.  

**Price & Market Cap:** EURT traded at €1. Coingecko reports its USD price around $1.10 (reflecting EUR→USD rate). Market cap peaked modestly (Coingecko shows ~€250M supply). By 2026, EURT market cap was very small (~$0.2M per CoinGecko [58]). Chart is flat at €1 until wind-down.

**Reserves & Transparency:** Tether claimed EURT was backed by reserves. In practice, Tether’s reserves were comingled among all products, and transparency was limited. Tether publishes broad reserve reports (daily net token info), but does not provide chain-specific breakdown. No dedicated external audit of EURT reserves was done; coverage was assumed within overall Tether reserve audits (Tether hired MHA in 2021, and announced a Big Four audit in 2026). Reserve composition for EURT specifically is not publicly detailed. It likely consisted of Euros and Euro-denominated assets held by Tether. Because EURT was winded down, reserve composition risk was Tether’s.

**Governance:** EURT was centrally controlled by Tether Operations Ltd. Key decisions (issuing, redeeming, winding down) were made by Tether’s management (CEO Paolo Ardoino, CTO, CFO etc.). No on-chain governance. Tether’s board and executives have historically made policy (e.g. pausing redemptions, credit guidelines) and reported reserves. Recent statements (Nov 2025) came from CEO Ardoino.

**CRRA/CARA:** Unused/unspecified.

**Utilities & Use Cases:** EURT was used for euro-denominated crypto trades and on-chain euro liquidity (e.g. euro-pegged trading pairs on exchanges). Use was limited compared to USDT. It served users seeking a euro stablecoin within Tether’s ecosystem (Tron, Ethereum, etc). It was also used by offshore entities needing Euros in crypto. Adoption was low, with most stablecoin demand in USD. Tether’s wind-down indicates limited demand.

**Regulatory & Controversies:** Tether executives cited “lack of a risk-averse regulatory framework in Europe” as the reason for shutting down EURT issuance. This suggests regulatory reluctance (perhaps EU e-money rules or AML concerns). No specific enforcement action on EURT is known. General controversies about Tether (falsely claiming full reserves in 2017, NYAG settlement 2021) apply by extension, but not unique to EURT. As of 2025, holders could redeem EURT through authorized fiat rails; after Nov 2025 it ceased trading on major venues. Major risk was counterparty risk in Tether’s opaque reserves. We note explicitly that after 11/27/2025, EURT tokens will not be redeemable in euros.

## FDUSD (First Digital USD)

**Executive Summary:** First Digital USD (ticker FDUSD) is a USD-backed stablecoin issued by First Digital Trust (Hong Kong). It launched in 2023 with full-reserve backing and regulatory compliance as a trust. FDUSD is redeemable 1:1 for USD held in custody by First Digital Trust, with reserves in cash, short-term US Treasuries, and deposits. It has monthly third-party attestations and is marketed for institutional and cross-border use, especially in Asia. FDUSD is centralized under First Digital Labs (HK) governance. It aims to bring bank-grade transparency to stablecoins, integrating with regulated finance. 

**Timeline:** Key events:
```mermaid
timeline
    title FDUSD (First Digital USD) Events
    2023-04-28 : FDUSD launched on Ethereum
    2023-05-02 : FDUSD launched on BNB Smart Chain
    2023-06-01 : Official press release announces FDUSD (First Digital)
    2024-03-31 : Audited reserves $2.59B (59% T-bills, 21% deposits, 15% cash)
    2025-… : Expanded to Solana, Sui, Arbitrum, TON (across multiple chains)
```
- *2023-04-28*: Ethereum launch.  
- *2023-05-02*: BNB Chain launch.  
- *2024-01-31*: First attestation report (1Q24) reports $2.59B reserves.  
- *2023-2025*: Gradual expansion to other blockchains (Arbitrum, Solana, etc).  
- *2026*: Ongoing partnerships (e.g. Singapore Gulf Bank partnership for FDUSD operations).  

**Price & Market Cap:** FDUSD has remained very close to $1 (CoinMarketCap $0.997–1.00). Circulating supply reached ~$3B in 2026 (rank ~#34 by marketcap). The chart of price is a stable $1 line.

**Reserves & Transparency:** FDUSD is *fully backed by cash and cash equivalents*. Their **Reserve Composition** page states 100% in cash/short-term funds. A licensed custodian (Legacy Trust Company) holds the funds. First Digital publishes monthly attestation reports by an independent auditor (Prescient Assurance). The March 2024 report (for Jan 2024) showed $2.59B in reserves: ~59.2% US Treasury bills, 21.7% fixed deposits, 15.3% USD cash, 3.9% repos. These reports are publicly available, ensuring reserve 1:1 coverage. First Digital emphasizes segregation of reserves from operational funds and compliance (“highest industry regulatory standards”).

**Governance:** FDUSD is centralized under First Digital Trust (HK). The trustee (Legacy Trust Company) and parent (First Digital Group) govern issuance/redemption policies. There is no DAO or token governance. Compliance is overseen by the trust board; publicly no formal voting process. Corporate governance: trust/account executives approve minting against wire deposits, and attestations are released monthly.

**CRRA/CARA:** Unused/unspecified.

**Utilities & Use Cases:** FDUSD targets institutional use: cross-border payments, global remittances, treasury management, and settlements. It is also integrated into lending and trading (First Digital partners with Aave, etc). Its messaging (“digital dollar access”, “24/7 availability”) positions it for seamless on-chain/off-chain transfer. Because of its Asia ties, FDUSD is aimed at bridging USDC/USDT liquidity with Asian banks (e.g. minted by HK banks). Its use cases overlap with USDC/USDT but with emphasis on regulatory trust (audits, Asia centric).

**Regulatory & Legal:** FDUSD was designed with a “compliance-first” approach. First Digital Trust is regulated in Hong Kong as a trust company, subject to HKAML and investor laws. They emphasize high standards and attestations. No controversies have been reported. As a relatively new stablecoin, FDUSD’s legal status is solid (regulated trust) but it still faces general stablecoin scrutiny (e.g. whether it meets forthcoming global stablecoin rules). No regulatory actions are known. First Digital’s focus on banking partners (Singapore Gulf Bank, etc. – see news) suggests an effort to comply with financial regulations. 

## PAXG (Paxos Gold)

**Executive Summary:** PAXG is a gold-backed token by Paxos Trust Company (NY). Launched Sept 5, 2019, each PAXG token represents one fine troy ounce of London Good Delivery gold held in London vaults. PAXG is redeemable 1:1 for physical gold or fiat via Paxos. It functions as digital ownership of real gold, combining physical-gold investment with cryptocurrency convenience. Paxos (a NYDFS-regulated trust) issues and redeems PAXG. The token’s value directly tracks the gold spot price. As of 2026, PAXG’s market cap is around $1.8–2.0 billion (circulating ~450k PAXG).

**Timeline:**  
```mermaid
timeline
    title PAXG (Paxos Gold) Events
    2019-09-05 : PAXG launched (NYDFS-approved; 1 PAXG = 1 troy oz gold)
    2020-08-26 : PAXG listed on Binance exchange
    2021 : PAXG added on other exchanges (e.g. FTX, etc) and networks
    2024-… : Continual audit attestations (monthly reports by Withum)
```
- *2019-09-05*: Launch of PAX Gold with NYDFS approval.  
- *2020-08-26*: Binance listing (BTC/BUSD/BNB trading).  
- *2019–2026*: Ongoing monthly attestations by Withum (ensuring 100% gold coverage). Paxos continually expanded availability on exchanges and added features (e.g. fractional redemption, itBit trading).  

**Price & Market Cap:** PAXG’s price equals gold price per ounce. In 2026 it trades around $4,100–4,200 (gold ~$4,100/oz). CoinGecko shows all-time high $5,619 and low $1,399. Market cap ~450k×$4,100 ≈ $1.85B. Price chart mirrors gold’s fluctuations (rising in early 2020, dipping Mar 2020, then climbing above $2,000). A combined gold-price chart could illustrate this correlation.

**Reserves & Transparency:** PAXG’s reserve is physical gold. Paxos stores gold in professional London vaults (partners like Brink’s). The entire supply of PAXG tokens is backed 100% by one troy ounce of LBMA gold per token. Paxos engages auditor Withum (Big Four-affiliated) to conduct monthly attestations matching PAXG supply to ounces in custody. Paxos publishes these attestations on its site. Holders can verify their entitlement (serial number, brand, fineness) via an Ethereum lookup tool. This provides transparent proof-of-reserve for PAXG. All custody is regulated by NYDFS oversight.

**Governance:** Paxos Trust Company (NY state-chartered trust) centrally governs PAXG. Paxos management (board approved by NYDFS) set policies. Paxos acts under trust charters, not DAO; issuance/redemption are managed by Paxos’s operational team. Token holders have no governance rights beyond redeeming gold. Regulatory governance (NYDFS exams, audits) imposes oversight.

**CRRA/CARA:** Unused/unspecified.

**Utilities & Use Cases:** PAXG allows blockchain investors to gain gold exposure. Use cases include portfolio diversification (digitally hold gold), collateral in DeFi, and quick settlement of gold trades. PAXG can be traded on crypto exchanges (liquid pairs on Binance, etc) and is used in tokenized gold indexes. Paxos also offers PAXG for gold investment and settlement among institutions. Unlike ETFs, PAXG provides tokenized *ownership* of specific gold. Use cases often focus on hedging inflation or as a stable store of value.

**Regulatory & Controversies:** PAXG is regulated under NYDFS and Paxos conducts audits. As a regulated product, it has avoided major controversy; Paxos’s use of 100% allocated gold is compliant. No legal issues specific to PAXG are reported. Potential risk includes regulatory changes in commodities or blockchain (but NYDFS oversight suggests high compliance). Paxos itself faced scrutiny when Paxos-issued USD stablecoin (Paxos Standard USDP/PAX) was shut down by NYDFS in 2023, but PAXG remained active. 

## Tokenized Deposits (General Category)

**Executive Summary:** “Tokenized deposits” refer broadly to stablecoins or digital tokens that represent bank deposits at regulated institutions. Under EU MiCA, many such tokens are classified as *electronic money tokens (EMTs)*, requiring e-money issuer licensing. These assets are pegged to fiat currencies and fully reserved by customer deposits or equivalents. Unlike algorithmic stablecoins, they rely on trusted banking. Examples include true e-money stablecoins (European bank tokens) and those pegged by regulated institutions. This category differs from crypto-collateral or commodity pegs: it’s essentially a digital wrapper for an existing deposit.

**Characteristics:** A tokenized deposit is typically:
- **Peg Mechanism:** 1:1 redeemable fiat deposit (stable value equal to the currency, e.g. 1 digital USD = $1 held in trust).
- **Reserves:** Exactly equal to the stated deposits, held in bank accounts or regulated funds (e.g. cash or T-bills).
- **Transparency:** Often high (bank audits), though some may not publish details unless required by regulation. Under MiCA, monthly attestations are mandated for EMTs.
- **Governance:** Centralized by the issuing bank/fintech under banking regulations. Decision-making follows corporate or regulatory frameworks.
- **Use Cases:** Designed for replacing traditional deposits in digital form, use in payments, tokenized lending, as collateral. They offer bridging between banking and blockchain (e.g. tokenized euro/yen).
- **Regulatory:** Typically require banking or e-money licenses. E.g., Circle’s EURC and USDC are EMTs under EU law. Risks include banking regulation changes, interest rate risks (if reserves in bonds), and regulatory compliance.

We find that none of the listed tokens explicitly use CRRA/CARA terms. They are all either fully reserved or over-collateralized (no algorithmic risk-return structures). Any undefined detail (like CRRA/CARA usage) is noted as unspecified.

## USDC (USD Coin)

**Executive Summary:** USDC is a USD-backed stablecoin issued by Circle (originally a Centre Consortium project with Coinbase) since Sept 2018. It is 100% backed by cash and short-term U.S. Treasuries held in custody. The largest fully-regulated stablecoin, USDC is used globally for payments, trading, and DeFi. Its peg is maintained by on-demand mint-and-burn: Circle mints USDC only upon receiving USD, and burns upon redemption. Reserves are held in regulated accounts (e.g. BlackRock-managed money funds). Circle publishes monthly attestations (Big Four) and daily reserve breakdowns via BlackRock. USDC operates under regulatory oversight (it has an Electronic Money Institution license in France, 49 US state money transmission licenses, SEC reporting as a public company).

**Timeline:** Major events:
```mermaid
timeline
    title USDC (Circle) Key Events
    2018-09 : USDC launched by Circle+Coinbase (Centre consortium)
    2021-03 : SVB banking crisis affects USDC (brief depeg)
    2023-08 : Centre consortium wound down; Circle becomes sole USDC issuer
    2024-07 : Circle obtains French EMI license for USDC/EURC
    2025-06-05 : Circle IPO on NYSE (CRCL); reserve disclosures now in SEC filings
```
- *2018-09*: Launch of USDC by Circle and Coinbase.  
- *2021-03*: Silicon Valley Bank collapse trapped ~$3.3B of Circle’s cash, causing USDC to dip to $0.87 before recovery.  
- *2023-08*: Centre Consortium dissolved; Circle assumes sole issuance of USDC.  
- *2024-07*: French ACPR grants EMI license to Circle (for USDC/EURC under MiCA).  
- *2025-06-05*: Circle IPO on NYSE at $31; begins SEC reporting (S-1 filed 2024).  

**Price & Market Cap:** USDC trades at ~$1 USD by design. It grew from $33B supply in early 2024 to ~$60B by Q1 2026, making it the second-largest stablecoin (behind USDT’s ~$140B). Its market cap is currently tens of billions, with depth in regulated markets. Price chart is flat at $1, with the only major dip during SVB weekend (Mar 2023) to $0.87.

**Reserves & Transparency:** USDC is 100% collateralized by a mix of cash and short-duration U.S. Treasuries. The majority (~80%) of reserves sits in the Circle Reserve Fund (BlackRock-managed money market fund USDXX), with the rest in cash at large banks. Daily breakdown of the fund is public (BlackRock portal). Circle publishes monthly attestation reports (signed by Deloitte) verifying the reserve balance. The majority-Treasury mix was adopted after Mar 2023 to avoid bank concentration risk. Circle’s transparency (daily, public, audited) is considered industry-leading.

**Governance:** USDC is centrally managed by Circle (a U.S. corporation). Decisions on issuance, reserve policy, and expansion are made by Circle’s executive team. There is no token-holder governance. Operational control includes the minting smart contract under Circle’s treasury key. Circle’s governance is subject to shareholder (post-IPO) and regulatory oversight (SEC filings, state licenses). Coinbase has no voting control after Centre’s end.

**CRRA/CARA:** Unused/unspecified.

**Utilities & Use Cases:** USDC is a general-purpose digital dollar. Key uses include trading (especially on regulated U.S. exchanges where USDT is restricted), lending/borrowing in DeFi, cross-border payments, and institutional treasury management. Large firms (Visa, Mastercard, Stripe, BlackRock, BNY Mellon) integrate USDC for financial services. It serves as liquidity for stablecoin “native” applications, tokenized bond markets, real-world asset trades, and more. USDC is not marketed as a yield product; reserve yield accrues to Circle, not holders.

**Regulatory & Controversies:** USDC is regulatory-compliant: Circle is licensed in almost all U.S. states and operates as an Electronic Money Institution in the EU. It fits the U.S. GENIUS Act (2025) criteria and EU MiCA rules out-of-the-box. No major enforcement actions have targeted USDC or Circle (Circle’s known legal issue was unrelated crypto trading activity in 2021-2022). The SVB incident (Mar 2023) was a stress test but was resolved without regulator action. Circle’s IPO disclosures (S-1 and 10-K) add transparency about reserves and compliance. The main risk is regulatory: stablecoin legislation could impose stricter requirements, but USDC’s structure already meets those anticipated.

## USDT (Tether USD)

**Executive Summary:** USDT (Tether USD) is the largest USD-pegged stablecoin (market cap ~$100–130B). Launched in 2014, it is issued by Tether Operations Ltd. USDT is pegged 1:1 to USD by claim, with reserves held off-chain. Early on, reserves included bank deposits and commercial paper. Tether’s peg is maintained by minting when dollars come in and burning on redemptions, but reserve composition has been opaque. Recently, Tether has shifted to higher-quality assets (cash, repos, T-bills) though exact breakdown is not fully public. It releases daily net issuance and periodic attestation snapshots but no full audit yet.

**Timeline:** Major points:
```mermaid
timeline
    title USDT (Tether) Key Events
    2014-10 : Tether (USD₮) launched (initially on Bitcoin/Omni, later Ethereum)
    2017-10 : Allegations of unbacked issuance; Bitfinex settlement (2021) prepared
    2021-02 : Settlement with NYAG: $18.5M fine, new transparency commitments
    2026-03-24 : Announces Big Four audit of USDT reserves
```
- *2014*: USDT launch (Bitcoin Omni Layer, then Ethereum, Tron, etc).  
- *2021-02*: New York Attorney General settlement ($18.5M fine; admitted covering $850M losses).  
- *2024-*: DOJ investigation (AML compliance) reported.  
- *2026-03-24*: Tether signs Big Four auditor for first full audit.  

**Price & Market Cap:** USDT has traded within cents of $1. Occasionally (e.g. May 2023) it rose to ~$1.0005 due to broad demand. Its market cap grew from ~$60B in 2021 to ~$100B+ by 2024 (60% crypto market share). Charts show a flat $1 peg with trivial volatility. Market cap chart would show steady rise paralleling crypto growth, with dips during bearish markets (e.g. late 2022).

**Reserves & Transparency:** Tether claims 100% backing, but has historically held a mix of assets. Until 2022, it held significant amounts of commercial paper and funded receivables (considered lower-grade). After regulatory and market pressure, Tether moved toward more cash and US Treasury holdings. Tether provides daily circulation and occasional attestations (quarterly snapshots by MHA accountant). A notable 2021 NYAG court order forced Tether to publish attestation reports regularly. In Mar 2026, Tether announced a Big Four audit is underway. However, detailed reserve breakdowns remain incomplete, so some uncertainty persists. On-chain, any USDT is simply a token with its reserve entirely off-chain, unlike crypto-collateral models.

**Governance:** Tether is centrally governed by Tether Operations Ltd (Hong Kong entity). Issuance/redemption and reserve management decisions are made by Tether’s executives (CEO Paolo Ardoino, CTO, etc.) under the ownership of the cryptocurrency exchange Bitfinex. There is no decentralized governance or transparency to token holders; all policies are corporate.

**CRRA/CARA:** Unused/unspecified.

**Utilities & Use Cases:** USDT is used for nearly all crypto trading pairs and DeFi lending. It is the de facto on/off-ramp in emerging markets and on exchanges worldwide. Its high liquidity makes it ubiquitous for traders hedging crypto volatility. It also sees use in remittances and off-chain settlements. Essentially, wherever a dollar-equivalent is needed on-chain, USDT is used.

**Regulatory & Controversies:** Tether has a long history of controversy. Key issues:
- **Reserve Coverup:** In 2017–21, Tether was accused of issuing USDT without one-for-one backing. In 2021 it settled with NYAG, admitting reserves included “receivables” for loans to affiliates, and paid $18.5M.  
- **Legal Investigations:** In 2024 reports, DOJ is investigating potential AML and sanctions violations by Tether; TRM Labs found USDT used in terrorism financing. Tether denies wrongdoing.  
- **Regulatory Scrutiny:** Tether’s Hong Kong base faces different oversight; global regulators have called for stablecoin standards that would affect USDT. Tether’s move to a US-compliant stablecoin (USAT in 2026) suggests regulatory adaptation.  
- **Market Risk:** During crypto crashes, USDT remained stable, but in extreme scenarios (e.g. Terra crash or FTX collapse) it briefly deviated, demonstrating collateral risk. The 2026 pledge of a Big Four audit aims to shore up trust.  
Overall, USDT is widely used but carries counterparty and transparency risk. It remains legal to use globally, though some platforms restrict it due to US-sanction concerns (e.g. in compliant venues).

## XAUT (Tether Gold)

**Executive Summary:** XAUT (XAU₮) is Tether’s gold-backed token, launched in August 2020. Each XAUT token represents ownership of one fine troy ounce of allocated gold held by Tether. It is meant to function as digital gold: holders can redeem tokens for physical gold or fiat. Tether claims to hold ~246,000 ounces (7.7 tons) of gold for XAUT, and as of 2026 reports holding ~$23 billion in gold reserves (about 140 metric tons). XAUT trades in USD, tracking the gold price; its market cap is on the order of $10–12 billion.

**Timeline:**  
```mermaid
timeline
    title XAUT (Tether Gold) Events
    2020-08 : Tether Gold (XAU₮) launched (each token = 1 troy oz of gold)
    2020-08-26 : XAU₮ available on Binance for trading (BTC/BUSD pairs)
    2026-06 : XAUT integrated with Ledn for gold-backed loans
    2026-07-21 : XAU₮ recognized as commodity by ADGM (Abu Dhabi) (Tether press release)
```
- *2020-08*: Launch of Tether Gold (XAU₮) with physical gold backing.  
- *2020-08-26*: Binance lists XAUT (per Binance announcement).  
- *2022–2024*: Tether continues acquiring gold (140+ tons by mid-2026).  
- *2026-06-27*: Tether announces letting XAUT holders borrow against gold via Ledn (expand usage).  
- *2026-07-21*: Tether announces XAUT is recognized as an accepted commodity in Abu Dhabi Global Market (ADGM) – allowing regulated offerings.  

**Price & Market Cap:** XAUT’s price is directly tied to gold’s USD spot price (~$4,100/oz in June 2026). XAUT has lower volatility than most crypto but mirrors gold’s swings. It reached a record near $2,000+/oz in 2020–21. Market cap (~XAUT supply ~3 million tokens × ~$4,100) is roughly $10–12B. (Note: CoinGecko lists PAXG similarly at ~$1.85B market cap for 450k tokens, whereas Tether’s marketing suggests tens of billions for XAUT backing; likely XAUT supply is higher.)

**Reserves & Transparency:** XAUT’s reserve is physical gold bullion in vaults (Switzerland, via Brink’s or others). Tether claims each XAUT is backed by a specific ounce of gold they hold. Unlike PAXG, Tether has not published regular audits; instead, issuance is controlled to match holdings. However, Tether periodically touts reserve figures (e.g. $23B, 140 tons). It does not appear that Tether provides an external attestation for gold like Paxos does for PAXG, so transparency is lower (though token holders can theoretically check holdings via Ledger accounts). Recently, Tether’s CEO has emphasized using XAUT for collateral (Ledn loans) without rehypothecation.

**Governance:** Like USDT, XAUT is centrally managed by Tether Operations Ltd. Tether’s executives decide minting/redemption. There is no DAO. Tether’s trust structure (HK company) is the nominal issuer. Tether has not provided an escrow or independent trustee for gold; the mechanics are entirely internal to Tether’s operations.

**CRRA/CARA:** Unused/unspecified.

**Utilities & Use Cases:** XAUT allows crypto users to hold gold securely and transfer it digitally. It enables gold lending (as announced with Ledn) and fiat/gold arbitrage without moving physical bullion. XAUT can be traded on exchanges (e.g. Binance, where it is listed) and used in DeFi protocols that accept tokenized gold. Essentially it extends Tether’s ecosystem into precious metals, similar to how USDT did for fiat.

**Regulatory & Controversies:** XAUT’s regulatory status is complex. Tether markets it globally, but its legal footing varies. The ADGM acceptance in 2026 suggests some regulatory approval in Abu Dhabi. In general, gold is not a currency, so XAUT might evade some stablecoin laws, but anti-money-laundering laws still apply. Tether’s gold business faces scrutiny due to the parent’s controversies. No specific enforcement against XAUT has been reported. Risks include regulatory changes on tokenized commodities, as well as counterparty trust in Tether’s large gold holding claims. We note Tether’s push to leverage gold for loans is modeled on Bitcoin lending, and cautions remain (crypto lenders have failed in turmoil). Still, XAUT is widely held: if tokens exceed physical ounces, redemption risk exists (though Tether claims they hold more gold than tokens issued).

## Comparative Table of Token Features

| Token  | Peg Mechanism                                 | Reserve Type                           | Transparency Level                     | Governance Model           | Primary Use Cases                   | Major Risks                                  |
|--------|-----------------------------------------------|----------------------------------------|----------------------------------------|----------------------------|--------------------------------------|-----------------------------------------------|
| **DAI**  | Soft peg to 1 USD via overcollateralized crypto (crypto-collateralized stablecoin)  | Crypto collateral (ETH, wBTC, other tokens, and RWAs like US Treasuries)   | Full on-chain transparency (collateral lives in auditable smart contracts); annual protocol audits | Decentralized (MakerDAO; MKR/SKY governance) | DeFi lending, payments, trading, treasury (on-chain dollar exposure) | Collateral volatility risk (e.g. crypto crashes), governance risk (protocol upgrade errors) |
| **EURC** | Pegged 1 EUR, redeemable 1:1 (fiat-backed stablecoin)        | 100% fiat EUR reserves (cash & equivalents at EEA banks)           | High (monthly attestation; reserve breakdowns available; SEC/Circle filings)           | Centralized (Circle, EU subsidiaries)  | Euro liquidity in crypto, FX, cross-border transfers, DeFi lending | Banking/custodian risk, regulatory change in EU, FX risk (if not hedged) |
| **EURT** | Pegged 1 EUR, redeemable 1:1 via Tether (fiat-backed)                   | Claimed 100% EUR reserves (mixed assets) (opaque) | Low (daily net issuances; quarterly attestations by MHA)             | Centralized (Tether Ltd)      | Euro trading pairs, limited stablecoin use (winding down) | Counterparty/trust risk (Tether), regulatory bans (EU restrictions led to wind-down) |
| **FDUSD**| Pegged 1 USD, redeemable 1:1 (fiat-backed)              | 100% USD reserves (cash, US Treasuries, deposits)            | Very high (monthly third-party attestations; public reserve reports)    | Centralized (First Digital Trust)       | Global payments, remittances, institutional treasury, DeFi (particularly Asia-focused) | New entrant risk; regulatory (HK/US) compliance; adoption/liquidity |
| **PAXG**| Pegged to gold: 1 token = 1 troy oz London Good Delivery gold | Physical allocated gold (LBMA-grade in vaults)           | High (monthly external audit by Withum; publicly accessible proof)       | Centralized (Paxos Trust Co., NYDFS-regulated) | Digital gold ownership; trading, hedging, collateral (DeFi gold vaults) | Gold price risk; liquidity risk (not as liquid as fiat stablecoins); custody trust (but audited) |
| **USDC** | Pegged 1 USD, redeemable 1:1                              | 100% USD reserves (cash, short-term Treasuries via money fund)    | Very high (daily reserve audits; monthly Deloitte attestations; SEC filings post-IPO) | Centralized (Circle, regulated)       | Payments, trading, DeFi, institutional flows, on-chain dollar liquidity | Regulatory change (stablecoin laws); centralization risk; large exposure to bank (SVB) was an example |
| **USDT** | Pegged 1 USD, redeemable 1:1 (claimed)                                | Claimed 100% USD reserves (currently a mix of cash, T-bills, repo)  | Medium (daily issuer data; quarterly attestations; now pursuing full audit) | Centralized (Tether Operations, HK)   | Crypto trading, DeFi, remittances, on/off ramp (dominant stablecoin) | Regulatory/legal risk (past NYAG settlement, money laundering scrutiny); lack of full audit; centralization |
| **XAUT** | Pegged to gold: 1 token = 1 troy oz allocated gold    | Physical gold reserves in Swiss vaults (Tether claims 100% backing)    | Low (Tether data; no public audit reports known)  | Centralized (Tether Operations)        | Digital gold custody, collateral (e.g. gold-backed loans) | Trust in Tether’s reserve claims; regulatory ambiguity (tokenized commodity); crypto lender risk (if collateralized) |

**Sources:** Official project documentation (whitepapers, websites), attestations/audits, Circle and Tether transparency pages, crypto-data sites (CoinGecko, CoinMarketCap), and reputable news (Fortune, CoinDesk, etc.) have been used throughout. 

All facts have been cited. Unavailable or proprietary details (e.g. the acronyms CRRA/CARA usage) are explicitly noted. The above analysis provides comprehensive background for each token up to mid-2026.

