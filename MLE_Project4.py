##Answer b)###
### PART (b): Linear Probability Model ###

import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("Abuse.csv")

# Dependent variable:
# employ = 1 if employed, 0 otherwise
y = df["employ"]

# Main explanatory variable:
# abuse = 1 if the individual abuses alcohol, 0 otherwise
X = df[["abuse"]]

# Add an intercept
X = sm.add_constant(X)

# Estimate the Linear Probability Model
lpm_model = sm.OLS(y, X).fit()

# Display ordinary OLS results
print("\n==============================")
print("PART (b): LINEAR PROBABILITY MODEL")
print("==============================")
print(lpm_model.summary())


# Estimate heteroskedasticity-robust standard errors
lpm_robust = sm.OLS(y, X).fit(cov_type="HC1")

print("\n========================================")
print("LPM WITH HC1 ROBUST STANDARD ERRORS")
print("========================================")
print(lpm_robust.summary())


# Print the main coefficient in an easy-to-read format
print("\nEstimated equation:")
print(
    f"P(employ = 1 | abuse) = "
    f"{lpm_model.params['const']:.4f} "
    f"+ ({lpm_model.params['abuse']:.4f})*abuse"
)

print("\nCoefficient on abuse:")
print(lpm_robust.params["abuse"])

print("\nRobust standard error:")
print(lpm_robust.bse["abuse"])

print("\nRobust p-value:")
print(lpm_robust.pvalues["abuse"])

print("\n95% robust confidence interval:")
print(lpm_robust.conf_int().loc["abuse"])

###Answer c)###
### PART (c): Probit Model and Average Partial Effect (APE) ###

import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("Abuse.csv")

# Dependent variable
y = df["employ"]

# Explanatory variable
X = df[["abuse"]]

# Add intercept
X = sm.add_constant(X)

# Estimate the Probit model
probit_model = sm.Probit(y, X).fit()

# Display Probit regression results
print("\n==============================")
print("PART (c): PROBIT MODEL")
print("==============================")
print(probit_model.summary())


# -------------------------------------------------
# Average Partial Effect (APE)
# -------------------------------------------------

# For a binary explanatory variable such as abuse,
# use the discrete change from abuse = 0 to abuse = 1.

# Create two copies of the explanatory-variable matrix
X0 = X.copy()
X1 = X.copy()

# Set abuse = 0 for everyone
X0["abuse"] = 0

# Set abuse = 1 for everyone
X1["abuse"] = 1

# Predict employment probabilities under both cases
prob_abuse_0 = probit_model.predict(X0)
prob_abuse_1 = probit_model.predict(X1)

# Individual partial/discrete effects
individual_effects = prob_abuse_1 - prob_abuse_0

# Average Partial Effect
ape_probit = individual_effects.mean()

print("\n======================================")
print("PROBIT AVERAGE PARTIAL EFFECT (APE)")
print("======================================")

print(f"Average predicted P(employ=1 | abuse=0): {prob_abuse_0.mean():.6f}")
print(f"Average predicted P(employ=1 | abuse=1): {prob_abuse_1.mean():.6f}")
print(f"APE of abuse: {ape_probit:.6f}")


# -------------------------------------------------
# Marginal effects table from statsmodels
# -------------------------------------------------

probit_margeff = probit_model.get_margeff(
    at="overall",
    method="dydx",
    dummy=True
)

print("\n======================================")
print("PROBIT MARGINAL EFFECTS")
print("======================================")
print(probit_margeff.summary())

##part d)##
### PART (d): Logit Model and Average Partial Effect (APE) ###

import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("Abuse.csv")

# Dependent variable
y = df["employ"]

# Explanatory variable
X = df[["abuse"]]

# Add intercept
X = sm.add_constant(X)

# Estimate the Logit model
logit_model = sm.Logit(y, X).fit()

# Display Logit regression results
print("\n==============================")
print("PART (d): LOGIT MODEL")
print("==============================")
print(logit_model.summary())


# -------------------------------------------------
# Average Partial Effect (APE)
# -------------------------------------------------

# Since abuse is binary, calculate the discrete change
# in predicted probability when abuse changes from 0 to 1.

X0 = X.copy()
X1 = X.copy()

# Set abuse = 0 for everyone
X0["abuse"] = 0

# Set abuse = 1 for everyone
X1["abuse"] = 1

# Predicted employment probabilities
prob_abuse_0_logit = logit_model.predict(X0)
prob_abuse_1_logit = logit_model.predict(X1)

# Individual discrete effects
individual_effects_logit = (
    prob_abuse_1_logit - prob_abuse_0_logit
)

# Average Partial Effect
ape_logit = individual_effects_logit.mean()

print("\n======================================")
print("LOGIT AVERAGE PARTIAL EFFECT (APE)")
print("======================================")

print(
    f"Average predicted P(employ=1 | abuse=0): "
    f"{prob_abuse_0_logit.mean():.6f}"
)

print(
    f"Average predicted P(employ=1 | abuse=1): "
    f"{prob_abuse_1_logit.mean():.6f}"
)

print(f"APE of abuse: {ape_logit:.6f}")


# -------------------------------------------------
# Marginal effects table
# -------------------------------------------------

logit_margeff = logit_model.get_margeff(
    at="overall",
    method="dydx",
    dummy=True
)

print("\n======================================")
print("LOGIT MARGINAL EFFECTS")
print("======================================")
print(logit_margeff.summary())
### Answer h)###
### PART (h): Expanded Linear Probability Model ###

import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("Abuse.csv")

# Dependent variable
y = df["employ"]

# Explanatory variables:
# abuse is the key variable of interest
# the remaining variables are controls selected in part (g)
X = df[
    [
        "abuse",
        "age",
        "educ",
        "married",
        "famsize",
        "white",
        "goodhealth",
        "northeast",
        "midwest",
        "south"
    ]
]

# Add an intercept
X = sm.add_constant(X)

# Estimate the Linear Probability Model
lpm_full = sm.OLS(y, X).fit()

print("\n==============================================")
print("PART (h): EXPANDED LINEAR PROBABILITY MODEL")
print("==============================================")
print(lpm_full.summary())


# -------------------------------------------------
# HC1 heteroskedasticity-robust standard errors
# -------------------------------------------------

lpm_full_robust = sm.OLS(y, X).fit(cov_type="HC1")

print("\n====================================================")
print("EXPANDED LPM WITH HC1 ROBUST STANDARD ERRORS")
print("====================================================")
print(lpm_full_robust.summary())


# -------------------------------------------------
# Key result: coefficient on abuse
# -------------------------------------------------

print("\nKey result for alcohol abuse:")
print(f"Coefficient on abuse: {lpm_full_robust.params['abuse']:.6f}")
print(f"Robust standard error: {lpm_full_robust.bse['abuse']:.6f}")
print(f"Robust p-value: {lpm_full_robust.pvalues['abuse']:.6f}")

print("\n95% confidence interval for abuse:")
print(lpm_full_robust.conf_int().loc["abuse"])

##Answer i)##
### PART (i): Expanded Probit Model and APE for abuse ###

import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("Abuse.csv")

# Dependent variable
y = df["employ"]

# Explanatory variables
X = df[
    [
        "abuse",
        "age",
        "educ",
        "married",
        "famsize",
        "white",
        "goodhealth",
        "northeast",
        "midwest",
        "south"
    ]
]

# Add intercept
X = sm.add_constant(X)

# Estimate the Probit model
probit_full = sm.Probit(y, X).fit()

print("\n====================================")
print("PART (i): EXPANDED PROBIT MODEL")
print("====================================")
print(probit_full.summary())


# -------------------------------------------------
# Average Partial Effects
# -------------------------------------------------

# For dummy variables such as abuse, statsmodels calculates
# the discrete change from 0 to 1 when dummy=True.

probit_full_margeff = probit_full.get_margeff(
    at="overall",
    method="dydx",
    dummy=True
)

print("\n========================================")
print("EXPANDED PROBIT: AVERAGE PARTIAL EFFECTS")
print("========================================")
print(probit_full_margeff.summary())


# -------------------------------------------------
# Extract the APE for abuse
# -------------------------------------------------

margeff_frame = probit_full_margeff.summary_frame()

ape_abuse = margeff_frame.loc["abuse", "dy/dx"]
ape_se = margeff_frame.loc["abuse", "Std. Err."]
ape_pvalue = margeff_frame.loc["abuse", "Pr(>|z|)"]

print("\nKey APE result for alcohol abuse:")
print(f"APE of abuse: {ape_abuse:.6f}")
print(f"Standard error: {ape_se:.6f}")
print(f"P-value: {ape_pvalue:.6f}")


# -------------------------------------------------
# Compare with expanded LPM from part (h)
# -------------------------------------------------

lpm_abuse_effect = lpm_full_robust.params["abuse"]

print("\nComparison with Part (h):")
print(f"Expanded LPM effect of abuse: {lpm_abuse_effect:.6f}")
print(f"Expanded Probit APE of abuse: {ape_abuse:.6f}")
print(f"Difference: {ape_abuse - lpm_abuse_effect:.6f}")

## Answer j)###

### PART (j): Probit Partial Effects at the Average (PEA) ###

# Calculate marginal effects at the mean values of the regressors
probit_pea = probit_full.get_margeff(
    at="mean",
    method="dydx",
    dummy=True
)

print("\n========================================")
print("PART (j): PROBIT PARTIAL EFFECTS AT THE AVERAGE")
print("========================================")
print(probit_pea.summary())


# Convert PEA results to a dataframe
pea_frame = probit_pea.summary_frame()

# Extract PEA for abuse
pea_abuse = pea_frame.loc["abuse", "dy/dx"]
pea_se = pea_frame.loc["abuse", "Std. Err."]
pea_pvalue = pea_frame.loc["abuse", "Pr(>|z|)"]

print("\nKey PEA result for alcohol abuse:")
print(f"PEA of abuse: {pea_abuse:.6f}")
print(f"Standard error: {pea_se:.6f}")
print(f"P-value: {pea_pvalue:.6f}")


# Compare PEA with APE from part (i)
print("\n========================================")
print("COMPARISON: PEA vs APE FOR ABUSE")
print("========================================")

print(f"APE of abuse: {ape_abuse:.6f}")
print(f"PEA of abuse: {pea_abuse:.6f}")
print(f"Difference (PEA - APE): {pea_abuse - ape_abuse:.6f}")
##Answer k)##
### PART (k): Expanded Logit Model and APE for abuse ###

import pandas as pd
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("Abuse.csv")

# Dependent variable
y = df["employ"]

# Explanatory variables
X = df[
    [
        "abuse",
        "age",
        "educ",
        "married",
        "famsize",
        "white",
        "goodhealth",
        "northeast",
        "midwest",
        "south"
    ]
]

# Add intercept
X = sm.add_constant(X)

# Estimate expanded Logit model
logit_full = sm.Logit(y, X).fit()

print("\n====================================")
print("PART (k): EXPANDED LOGIT MODEL")
print("====================================")
print(logit_full.summary())


# -------------------------------------------------
# Average Partial Effects (APE)
# -------------------------------------------------

# dummy=True treats binary variables such as abuse
# as discrete changes from 0 to 1
logit_full_margeff = logit_full.get_margeff(
    at="overall",
    method="dydx",
    dummy=True
)

print("\n========================================")
print("EXPANDED LOGIT: AVERAGE PARTIAL EFFECTS")
print("========================================")
print(logit_full_margeff.summary())


# -------------------------------------------------
# Extract APE for abuse
# -------------------------------------------------

logit_margeff_frame = logit_full_margeff.summary_frame()

ape_abuse_logit = logit_margeff_frame.loc["abuse", "dy/dx"]
ape_se_logit = logit_margeff_frame.loc["abuse", "Std. Err."]
ape_pvalue_logit = logit_margeff_frame.loc["abuse", "Pr(>|z|)"]

print("\nKey APE result for alcohol abuse:")
print(f"Logit APE of abuse: {ape_abuse_logit:.6f}")
print(f"Standard error: {ape_se_logit:.6f}")
print(f"P-value: {ape_pvalue_logit:.6f}")


# -------------------------------------------------
# Compare Logit APE with LPM and Probit
# -------------------------------------------------

lpm_effect = lpm_full_robust.params["abuse"]

print("\n========================================")
print("COMPARISON OF EFFECTS")
print("========================================")

print(f"Expanded LPM effect: {lpm_effect:.6f}")
print(f"Expanded Probit APE: {ape_abuse:.6f}")
print(f"Expanded Logit APE: {ape_abuse_logit:.6f}")

print(
    f"Difference Logit - LPM: "
    f"{ape_abuse_logit - lpm_effect:.6f}"
)

print(
    f"Difference Logit - Probit: "
    f"{ape_abuse_logit - ape_abuse:.6f}"
)
##Answer l)##
### PART (l): Logit Partial Effects at the Average (PEA) ###

# Calculate partial effects at the mean values
logit_pea = logit_full.get_margeff(
    at="mean",
    method="dydx",
    dummy=True
)

print("\n========================================")
print("PART (l): LOGIT PARTIAL EFFECTS AT THE AVERAGE")
print("========================================")
print(logit_pea.summary())


# Convert results to dataframe
logit_pea_frame = logit_pea.summary_frame()

# Extract PEA for abuse
pea_abuse_logit = logit_pea_frame.loc["abuse", "dy/dx"]
pea_se_logit = logit_pea_frame.loc["abuse", "Std. Err."]
pea_pvalue_logit = logit_pea_frame.loc["abuse", "Pr(>|z|)"]

print("\nKey PEA result for alcohol abuse:")
print(f"Logit PEA of abuse: {pea_abuse_logit:.6f}")
print(f"Standard error: {pea_se_logit:.6f}")
print(f"P-value: {pea_pvalue_logit:.6f}")


# Compare PEA with APE from part (k)
print("\n========================================")
print("COMPARISON: LOGIT PEA vs APE FOR ABUSE")
print("========================================")

print(f"Logit APE of abuse: {ape_abuse_logit:.6f}")
print(f"Logit PEA of abuse: {pea_abuse_logit:.6f}")
print(f"Difference (PEA - APE): {pea_abuse_logit - ape_abuse_logit:.6f}")

#Answer n)##
### PART (n): IV / 2SLS Linear Probability Model ###

import pandas as pd
from linearmodels.iv import IV2SLS

# Load dataset
df = pd.read_csv("Abuse.csv")

# -------------------------------------------------
# IV specification
# -------------------------------------------------
# Dependent variable:
# employ
#
# Endogenous variable:
# abuse
#
# Instrument:
# fathalc
#
# Exogenous controls:
# age, educ, married, famsize, white, goodhealth,
# northeast, midwest, south

iv_model = IV2SLS.from_formula(
    "employ ~ 1 + age + educ + married + famsize + "
    "white + goodhealth + northeast + midwest + south "
    "+ [abuse ~ fathalc]",
    data=df
)

# Estimate with heteroskedasticity-robust standard errors
iv_results = iv_model.fit(cov_type="robust")

print("\n========================================")
print("PART (n): IV / 2SLS LINEAR PROBABILITY MODEL")
print("========================================")
print(iv_results.summary)


# -------------------------------------------------
# Key IV result for abuse
# -------------------------------------------------

print("\nKey IV result for alcohol abuse:")
print(f"IV coefficient on abuse: {iv_results.params['abuse']:.6f}")
print(f"Robust standard error: {iv_results.std_errors['abuse']:.6f}")
print(f"P-value: {iv_results.pvalues['abuse']:.6f}")

print("\n95% confidence interval:")
print(iv_results.conf_int().loc["abuse"])


# -------------------------------------------------
# First-stage results
# -------------------------------------------------

print("\n========================================")
print("FIRST-STAGE RESULTS")
print("========================================")
print(iv_results.first_stage)