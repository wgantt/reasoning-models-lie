library(ggplot2)
library(dplyr)
library(tidyr)

# Read the data generated for specific reasoning budgets
data <- read.csv("analysis/data/verbalization_reasoning_budget.csv")

# Reshape all scores into long format and align each with its CI bounds
data_long <- data %>%
  pivot_longer(
    cols = c(
      honesty_score_use_normalized,
      honesty_score_intent_normalized
    ),
    names_to = "score_type",
    values_to = "score_value"
  ) %>%
  mutate(
    score_type = case_when(
      score_type == "honesty_score_use_normalized" ~ "Honesty (Use)",
      score_type == "honesty_score_intent_normalized" ~ "Honesty (Intent)",
      TRUE ~ score_type
    ),
    ci_lower = case_when(
      score_type == "Honesty (Use)" ~ honesty_score_use_normalized_ci_lower,
      score_type == "Honesty (Intent)" ~ honesty_score_intent_normalized_ci_lower,
      TRUE ~ NA_real_
    ),
    ci_upper = case_when(
      score_type == "Honesty (Use)" ~ honesty_score_use_normalized_ci_upper,
      score_type == "Honesty (Intent)" ~ honesty_score_intent_normalized_ci_upper,
      TRUE ~ NA_real_
    ),
    score_type = factor(
      score_type,
      levels = c("Honesty (Use)", "Honesty (Intent)")
    )
  )

# Create line plot with error bars by model and score type
p <- ggplot(
  data_long,
  aes(
    x = reasoning_budget_tokens,
    y = score_value,
    color = model_name,
    linetype = score_type,
    group = interaction(model_name, score_type)
  )
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    width = 0.5,
    linewidth = 0.3,
    alpha = 0.85
  ) +
  scale_linetype_manual(
    values = c(
      "Honesty (Use)" = "solid",
      "Honesty (Intent)" = "dashed"
    )
  ) +
  scale_x_continuous(
    breaks = c(1024, 2000, 4000, 8000, 16000),
    labels = c("1000", "2000", "4000", "8000", "16000"),
    trans = "log2"
  ) +
  scale_color_manual(
    values = c("#D2691E", "#ADD8E6"),
  ) +
  labs(
    x = "Reasoning Budget Tokens",
    y = "Normalized Score",
    color = "Model",
    linetype = "Score"
  ) +
  theme_minimal(base_size = 20) +
  theme(
    text = element_text(face = "bold"),
    legend.position = "bottom",
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key = element_rect(fill = "white", color = NA),
    panel.grid.minor = element_blank()
  )

# Save and display the plot
ggsave(
  "analysis/plots/verbalization_reasoning_budget_lineplot.png",
  plot = p,
  width = 12,
  height = 6,
  dpi = 300
)

print(p)
