library(ggplot2)
library(dplyr)
library(tidyr)

# Read the data
data <- read.csv("analysis/data/changed_to_hinted.csv")

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

# Baseline change to hint percentages
baseline_change_to_hint <- c(
  "GPQA-Diamond" = 33.33,
  "MMLU-Pro" = 12.83
)

# Add color and alpha to data
data <- data %>%
  mutate(
    bar_color = model_colors[model_name],
    bar_alpha = setting_alphas[setting],
    baseline = baseline_change_to_hint[dataset],
    label_text = ifelse(
      !is.na(changed_to_hinted_p_value) & changed_to_hinted_p_value < 0.05,
      sprintf("%.1f*", changed_to_hinted_percentage),
      sprintf("%.1f", changed_to_hinted_percentage)
    )
  )

# Create the plot
p <- ggplot(data, aes(x = model_setting, y = changed_to_hinted_percentage)) +
  geom_bar(
    aes(fill = model_name, alpha = setting),
    stat = "identity",
    position = "dodge",
    width = 0.95
  ) +
  geom_hline(
    aes(yintercept = baseline),
    linetype = "dashed",
    color = "gray60",
    size = 1
  ) +
  geom_errorbar(
    aes(
      ymin = changed_to_hinted_percentage_ci_lower,
      ymax = changed_to_hinted_percentage_ci_upper
    ),
    width = 0.3,
    size = 0.5,
    color = "gray50"
  ) +
  geom_text(
    aes(label = label_text),
    vjust = -0.5,
    size = 7.5,
    fontface = "bold",
    position = position_dodge(width = 0.95)
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
  labs(
    y = "% Change to Hint",
    x = NULL
  ) +
  scale_y_continuous(limits = c(0, 112.5), breaks = c(0, 25, 50, 75, 100)) +
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
    legend.text = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 27, face = "bold"),
    panel.spacing = unit(1, "lines")
  ) +
  guides(
    fill = guide_legend(order = 1),
    alpha = guide_legend(order = 2, override.aes = list(fill = "gray"))
  )

# Adjust y-axis labels to show only on leftmost subplots
p <- p + theme(
  strip.text.y = element_text(angle = 270, size = 24, face = "bold")
)

# Save the plot
ggsave(
  "analysis/plots/hint_usage_barplot.png",
  plot = p,
  width = 20,
  height = 8,
  dpi = 300
)

# Display the plot
print(p)
