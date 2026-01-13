library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpattern)

# Read the data
data <- read.csv("analysis/data/verbalization.csv")

# Recode hint type labels for display
data <- data %>%
  mutate(
    hint_type = case_when(
      hint_type == "Unethical Information" ~ "Unethical Info",
      hint_type == "Sycophancy V1" ~ "Sycophancy",
      TRUE ~ hint_type
    ),
    dataset = case_when(
      dataset == "GPQA" ~ "GPQA-Diamond",
      dataset == "MMLU_PRO" ~ "MMLU-Pro",
      TRUE ~ dataset
    )
  )

# Create a combined factor for model and setting
data <- data %>%
  mutate(
    model_setting = paste(model_name, setting, sep = "_"),
    # Create factor levels for ordering bars
    model_setting = factor(model_setting, levels = c(
      "Claude 4.5 Haiku_Correct Hint",
      "Claude 4.5 Haiku_Incorrect Hint",
      "Kimi K2 Thinking_Correct Hint",
      "Kimi K2 Thinking_Incorrect Hint",
      "Qwen 3 Next_Correct Hint",
      "Qwen 3 Next_Incorrect Hint"
    )),
    # Order hint types
    hint_type = factor(hint_type, levels = c(
      "Grader Hacking",
      "Metadata",
      "Unethical Info",
      "Sycophancy"
    )),
    # Order datasets
    dataset = factor(dataset, levels = c("GPQA-Diamond", "MMLU-Pro"))
  )

# Define colors for models
model_colors <- c(
  "Claude 4.5 Haiku" = "#D2691E",  # Light orange
  "Kimi K2 Thinking" = "#ADD8E6",   # Light blue
  "Qwen 3 Next" = "#90EE90"      # Light green
)

# Define alpha values for settings (solid for Correct, shaded for Incorrect)
setting_alphas <- c(
  "Correct Hint" = 1.0,
  "Incorrect Hint" = 0.5
)

# Add color and alpha to data
data <- data %>%
  mutate(
    bar_color = model_colors[model_name],
    bar_alpha = setting_alphas[setting],
    scores_match = faithfulness_score_normalized == honesty_score_normalized
  )

# Reshape data to long format for both scores
data_long <- data %>%
  pivot_longer(
    cols = c(faithfulness_score_normalized, honesty_score_normalized),
    names_to = "score_type",
    values_to = "score_value"
  ) %>%
  mutate(
    score_type = case_when(
      score_type == "faithfulness_score_normalized" ~ "Faithfulness",
      score_type == "honesty_score_normalized" ~ "Honesty",
      TRUE ~ score_type
    ),
    score_type = factor(score_type, levels = c("Faithfulness", "Honesty"))
  )

# Add CI bounds to the long format
data_long <- data_long %>%
  mutate(
    ci_lower = ifelse(score_type == "Faithfulness", 
                      faithfulness_score_normalized_ci_lower, 
                      honesty_score_normalized_ci_lower),
    ci_upper = ifelse(score_type == "Faithfulness", 
                      faithfulness_score_normalized_ci_upper, 
                      honesty_score_normalized_ci_upper),
    show_label = !(score_type == "Honesty" & scores_match)
  )

# Create the plot
p <- ggplot(data_long, aes(x = model_setting, y = score_value)) +
  geom_bar_pattern(
    aes(fill = model_name, alpha = setting,
        pattern = score_type,
        pattern_fill = score_type),
    stat = "identity",
    position = "identity",
    width = 0.9,
    pattern_color = "white",
    pattern_density = 0.3,
    pattern_spacing = 0.060,
    pattern_angle = 45,
    pattern_key_scale_factor = 1
  ) +
  geom_errorbar(
    data = subset(data_long, score_type == "Honesty"),
    aes(
      ymin = ci_lower,
      ymax = ci_upper,
      group = interaction(model_setting, score_type)
    ),
    width = 0.2,
    size = 0.8,
    color = "gray50",
    position = "identity"
  ) +
  geom_text(
    data = subset(data_long, show_label),
    aes(label = ifelse(score_value == 100, "100", sprintf("%.1f", score_value)),
        group = interaction(model_setting, score_type),
        vjust = ifelse(score_type == "Honesty", -0.25, -1.25),
        color = score_type),
    size = 7.5,
    fontface = "bold",
    position = "identity"
  ) +
  scale_color_manual(
    values = c("Faithfulness" = "black", "Honesty" = "black"),
    guide = "none"
  ) +
  facet_grid(
    dataset ~ hint_type,
    scales = "free_x",
    space = "free_x"
  ) +
  scale_fill_manual(
    name = "Model",
    values = model_colors
  ) +
  scale_alpha_manual(
    name = "Setting",
    values = setting_alphas,
    labels = c("Correct Hint", "Incorrect Hint")
  ) +
  scale_pattern_manual(
    name = "Score Type",
    values = c("Faithfulness" = "none", "Honesty" = "crosshatch"),
    labels = c("Faithfulness", "Honesty")
  ) +
  scale_pattern_fill_manual(
    name = "Score Type",
    values = c("Faithfulness" = NA, "Honesty" = "white"),
    labels = c("Faithfulness", "Honesty")
  ) +
  labs(
    y = "Normalized Score",
    x = NULL
  ) +
  scale_y_continuous(limits = c(0, 125), breaks = c(0, 25, 50, 75, 100)) +
  theme_minimal(base_size = 24) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 27, face = "bold"),
    axis.title.y = element_text(size = 30, face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    strip.text = element_text(size = 30, face = "bold"),
    strip.text.x = element_text(size = 30, face = "bold"),
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.text = element_text(size = 27, face = "bold"),
    legend.title = element_text(size = 27, face = "bold"),
    panel.spacing = unit(1, "lines")
  ) +
  guides(
    fill = guide_legend(order = 1, override.aes = list(pattern = "none")),
    alpha = guide_legend(order = 2, override.aes = list(fill = "gray", pattern = "none")),
    pattern = "none",
    pattern_fill = "none"
  )

# Adjust y-axis labels to show only on leftmost subplots
p <- p + theme(
  strip.text.y = element_text(angle = 270, size = 24, face = "bold")
)

# Save the plot
ggsave(
  "analysis/plots/verbalization_barplot.png",
  plot = p,
  width = 20,
  height = 8,
  dpi = 300
)

# Display the plot
print(p)
