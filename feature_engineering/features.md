**Transaction-level**
- amount: Raw transfer amount.
- log_amount: Natural log of (1 + amount); stabilizes scale and reduces skew.
- hour: Hour of day of the transaction (0–23, in UTC unless you changed tz).
- dow: Day of week (0=Monday … 6=Sunday).
- is_weekend: 1 if Saturday or Sunday, else 0.
- currency: Transaction currency code (one-hot encoded).
- same_currency: 1 if currency equals "CHF" (as coded), else 0. Adjust logic if multi-currency.

**Sender behavior/history**
- s_prior_send_count: Number of prior transactions sent by this sender before this one (0 means first send).
- s_cum_amount: Cumulative sum of amounts the sender sent before this transaction.
- s_prior_avg_amount: Sender’s average sent amount up to, but not including, this transaction (s_cum_amount / s_prior_send_count; 0 if no history).
- s_amt_dev_from_avg: Current amount minus sender’s prior average amount (positive = unusually large vs. their norm).
- s_account_age_days: Days from sender account open_date to the transaction date (clipped at 0).
- s_is_active: Sender account active flag as 1/0 (boolean cast).
- s_cust_age_cat: Sender customer age bucket (categorical; one-hot encoded).
- s_cust_gender: Sender customer gender (categorical; one-hot encoded).
- s_cust_city: Sender customer city (categorical; one-hot encoded).

**Receiver behavior/history**
- r_prior_recv_count: Number of prior transactions received by this receiver before this one.
- r_account_age_days: Days from receiver account open_date to the transaction date (clipped at 0).
- r_is_active: Receiver account active flag as 1/0.
- r_cust_age_cat: Receiver customer age bucket (categorical; one-hot encoded).
- r_cust_gender: Receiver customer gender (categorical; one-hot encoded).
- r_cust_city: Receiver customer city (categorical; one-hot encoded).

**Pair/relationship features**
- prior_pair_txn_count: Number of prior transactions from this sender to this receiver (directional) before this one.
- prior_rev_pair_txn_count: Number of prior transactions in the reverse direction (receiver → sender) before this one.
- same_customer: 1 if sender and receiver share the same customer_id, else 0.
- same_city: 1 if sender and receiver customer cities match, else 0.

**Internal helper keys (created but dropped before modeling)**
- pair_key: String "sender_account_id|receiver_account_id" used to compute pair counts.
- rev_pair_key: Reverse-direction key "receiver_account_id|sender_account_id".

**Target (not used as a feature)**
- is_fraud: Binary label used for training/validation.