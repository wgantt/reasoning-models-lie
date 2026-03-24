library(ggplot2)
library(dplyr)
library(tidyr)

# Read the data generated for specific reasoning budgets
data <- read.csv("analysis/data/verbalization_reasoning_budget.csv")

# Reshape both scores into long format and align each with its CI bounds
data_long <- data %>%
  pivot_longer(
    cols = c(faithfulness_score_normalized, honesty_score_use_normalized),
    names_to = "score_type",
    values_to = "score_value"
  ) %>%
  mutate(
    score_type = case_when(
      score_type == "faithfulness_score_normalized" ~ "Faithfulness",
      score_type == "honesty_score_use_normalized" ~ "Honesty (Use)",
      TRUE ~ score_type
    ),
    ci_lower = case_when(
      score_type == "Faithfulness" ~ faithfulness_score_normalized_ci_lower,
      score_type == "Honesty (Use)" ~ honesty_score_use_normalized_ci_lower,
      TRUE ~ NA_real_
    ),
    ci_upper = case_when(
      score_type == "Faithfulness" ~ faithfulness_score_normalized_ci_upper,
      score_type == "Honesty (Use)" ~ honesty_score_use_normalized_ci_upper,
      TRUE ~ NA_real_
    ),
    score_type = factor(score_type, levels = c("Faithfulness", "Honesty (Use)"))
  )

# Create line plot with error bars for each score
p <- ggplot(
  data_long,
  aes(x = reasoning_budget_tokens, y = score_value, color = score_type)
) +
  geom_line(aes(group = score_type), linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    width = 250,
    linewidth = 0.8,
    alpha = 0.85
  ) +
  scale_color_manual(
    values = c(
      "Faithfulness" = "#1b9e77",
      "Honesty (Use)" = "#d95f02"
    )
  ) +
  scale_x_continuous(breaks = sort(unique(data$reasoning_budget_tokens))) +
  labs(
    x = "Reasoning Budget Tokens",
    y = "Normalized Score",
    color = "Score"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

# Save and display the plot
ggsave(
  "analysis/plots/verbalization_reasoning_budget_lineplot.png",
  plot = p,
  width = 10,
  height = 6,
  dpi = 300
)

print(p)
