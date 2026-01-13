library(ggplot2)
library(dplyr)
library(tidyr)

# Read the data
data <- read.csv("analysis/data/accuracy.csv")

# Separate baseline data from experimental data
baseline_data <- data %>%
  filter(hint_type == "Baseline") %>%
  select(model_name, dataset, accuracy, accuracy_ci_lower, accuracy_ci_upper)

# Filter out baseline rows from main data
experimental_data <- data %>%
  filter(hint_type != "Baseline")

# Recode hint type labels and model names for display
experimental_data <- experimental_data %>%
  mutate(
    hint_type = case_when(
      hint_type == "Unethical Information" ~ "Unethical Info",
      hint_type == "Sycophancy V1" ~ "Sycophancy",
      TRUE ~ hint_type
    ),
    dataset = case_when(
      dataset == "GPQA" ~ "GPQA",
      dataset == "MMLU_PRO" ~ "MMLU-Pro",
      TRUE ~ dataset
    ),
    model_name = case_when(
      model_name == "Claude 4.5 Haiku" ~ "Claude",
      model_name == "Kimi K2 Thinking" ~ "Kimi",
      model_name == "Qwen 3 Next" ~ "Qwen",
      TRUE ~ model_name
    )
  )

# Update baseline data with same dataset and model name recoding
baseline_data <- baseline_data %>%
  mutate(
    dataset = case_when(
      dataset == "GPQA" ~ "GPQA",
      dataset == "MMLU_PRO" ~ "MMLU-Pro",
      TRUE ~ dataset
    ),
    model_name = case_when(
      model_name == "Claude 4.5 Haiku" ~ "Claude",
      model_name == "Kimi K2 Thinking" ~ "Kimi",
      model_name == "Qwen 3 Next" ~ "Qwen",
      TRUE ~ model_name
    )
  )

# Create factors for ordering
experimental_data <- experimental_data %>%
  mutate(
    # Create factor for setting
    setting = factor(setting, levels = c(
      "Correct Hint",
      "Incorrect Hint"
    )),
    # Order model names
    model_name = factor(model_name, levels = c(
      "Claude",
      "Kimi",
      "Qwen"
    )),
    # Order hint types
    hint_type = factor(hint_type, levels = c(
      "Grader Hacking",
      "Metadata",
      "Unethical Info",
      "Sycophancy"
    )),
    # Order datasets
    dataset = factor(dataset, levels = c("GPQA", "MMLU-Pro"))
  )

# Order baseline data factors
baseline_data <- baseline_data %>%
  mutate(
    model_name = factor(model_name, levels = c(
      "Claude",
      "Kimi",
      "Qwen"
    )),
    dataset = factor(dataset, levels = c("GPQA", "MMLU-Pro"))
  )

# Convert accuracy to percentage
experimental_data <- experimental_data %>%
  mutate(
    accuracy = accuracy * 100,
    accuracy_ci_lower = accuracy_ci_lower * 100,
    accuracy_ci_upper = accuracy_ci_upper * 100
  )

baseline_data <- baseline_data %>%
  mutate(
    accuracy = accuracy * 100,
    accuracy_ci_lower = accuracy_ci_lower * 100,
    accuracy_ci_upper = accuracy_ci_upper * 100
  )

# Define colors for models
model_colors <- c(
  "Claude" = "#D2691E",  # Light orange
  "Kimi" = "#ADD8E6",     # Light blue
  "Qwen" = "#90EE90"      # Light green
)

# Define alpha values for settings (solid for Correct, shaded for Incorrect)
setting_alphas <- c(
  "Correct Hint" = 1.0,
  "Incorrect Hint" = 0.5
)

# Expand baseline data to include all settings for ribbon plotting
baseline_expanded <- baseline_data %>%
  crossing(setting = factor(c("Correct Hint", "Incorrect Hint"), 
                           levels = c("Correct Hint", "Incorrect Hint"))) %>%
  mutate(
    xmin = as.numeric(setting) - 0.5,
    xmax = as.numeric(setting) + 0.5
  )

# Create the plot
p <- ggplot(experimental_data, aes(x = setting, y = accuracy)) +
  geom_bar(
    aes(fill = model_name, alpha = setting),
    stat = "identity",
    position = "dodge",
    width = 0.8
  ) +
  # Add shaded region for baseline CI bounds
  geom_rect(
    data = baseline_expanded,
    aes(xmin = xmin, xmax = xmax, ymin = accuracy_ci_lower, ymax = accuracy_ci_upper),
    fill = "gray80",
    alpha = 0.4,
    inherit.aes = FALSE
  ) +
  # Add baseline line
  geom_hline(
    data = baseline_data,
    aes(yintercept = accuracy),
    linetype = "dashed",
    size = 1,
    color = "black"
  ) +
  geom_errorbar(
    aes(
      ymin = accuracy_ci_lower,
      ymax = accuracy_ci_upper
    ),
    width = 0.25,
    size = 0.5,
    color = "gray50"
  ) +
  geom_text(
    aes(label = sprintf("%.1f", accuracy)),
    vjust = -0.8,
    size = 8.5,
    fontface = "bold",
    position = position_dodge(width = 0.8)
  ) +
  facet_grid(
    dataset + model_name ~ hint_type,
    scales = "free_x",
    space = "free_x",
    labeller = labeller(
      dataset = label_value,
      model_name = label_value,
      hint_type = label_value
    )
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
    y = "Accuracy (%)",
    x = NULL
  ) +
  scale_y_continuous(limits = c(0, 120), breaks = c(0, 25, 50, 75, 100)) +
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
    strip.text.y.left = element_text(angle = 0, size = 24, face = "bold"),
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.text = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 27, face = "bold"),
    panel.spacing = unit(1, "lines"),
    strip.placement = "outside"
  ) +
  guides(
    fill = guide_legend(order = 1),
    alpha = guide_legend(order = 2, override.aes = list(fill = "gray"))
  )
# Save the plot
ggsave(
  "analysis/plots/accuracy_barplot.png",
  plot = p,
  width = 20,
  height = 16,
  dpi = 300
)

# Display the plot
print(p)
