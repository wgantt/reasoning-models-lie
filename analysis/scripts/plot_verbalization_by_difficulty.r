library(ggplot2)
library(dplyr)
library(tidyr)

# Example counts for GPQA - incorrect hints
# - E: 105
# - M: 43
# - H: 20
# - HH: 30
# Read the data
data <- read.csv("analysis/data/verbalization_scores_by_difficulty.csv")

# Set ordered factors
data <- data %>%
  mutate(
    difficulty = factor(difficulty, levels = c("EASY", "MEDIUM", "HARD", "HARDEST")),
    hint_type = factor(
      hint_type,
      levels = c("grader_hacking", "metadata", "unethical_information", "sycophancy"),
      labels = c("Grader Hacking", "Metadata", "Unethical Info", "Sycophancy")
    ),
    hint_correctness = factor(
      hint_correctness,
      levels = c("correct", "incorrect"),
      labels = c("Correct Hint", "Incorrect Hint")
    ),
    model_name = factor(
      model_name,
      levels = c("Claude 4.5 Haiku", "Kimi-K2.5", "Qwen3-Next-80B-A3B-Thinking"),
      labels = c("Claude 4.5 Haiku", "Kimi-K2.5", "Qwen3-Next")
    )
  )

# Metric columns and their display labels (only the three requested)
metrics_cols <- c(
  "hint_present_true_proportion",
  "hint_use_true_proportion",
  "hint_intent_true_proportion"
)
metric_labels <- c(
  "hint_present_true_proportion" = "Hint Present",
  "hint_use_true_proportion"     = "Hint Use",
  "hint_intent_true_proportion"  = "Hint Intent"
)

# Reshape to long format, drop rows with no verbalization data (NA proportion)
data_long <- data %>%
  pivot_longer(
    cols = all_of(metrics_cols),
    names_to = "metric",
    values_to = "proportion"
  ) %>%
  mutate(
    metric = factor(metric_labels[metric], levels = unname(metric_labels))
  ) %>%
  filter(!is.na(proportion))

# Colorblind-friendly palette (Dark2)
metric_colors <- c(
  "Hint Present" = "#1b9e77",
  "Hint Use"     = "#d95f02",
  "Hint Intent"  = "#7570b3"
)

# One plot per dataset × hint_correctness combination (4 plots total)
for (ds in c("MMLU-Pro", "GPQA-Diamond")) {
  for (hc in c("Correct Hint", "Incorrect Hint")) {
    data_sub <- data_long %>%
      filter(dataset == ds, hint_correctness == hc)

    if (nrow(data_sub) == 0) next

    # Build x-axis labels: "EASY\n(n)" using n_examples_in_bin for this dataset
    # (bin sizes are the same across all model/hint combinations for a given dataset)
    bin_sizes <- data %>%
      filter(dataset == ds) %>%
      distinct(difficulty, n_examples_in_bin) %>%
      arrange(difficulty)
    difficulty_short_labels <- c(
      "EASY" = "E",
      "MEDIUM" = "M",
      "HARD" = "H",
      "HARDEST" = "HH"
    )
    x_labels <- setNames(
      difficulty_short_labels[as.character(bin_sizes$difficulty)],
      bin_sizes$difficulty
    )

    y_breaks <- c(0, 0.25, 0.5, 0.75, 1.0)
    grid_data <- data_sub %>%
      distinct(model_name, hint_type) %>%
      tidyr::crossing(y_value = y_breaks)

    p <- ggplot(data_sub, aes(x = difficulty, y = proportion, color = metric, group = metric)) +
      geom_segment(
        data = grid_data,
        aes(x = "EASY", xend = "HARDEST", y = y_value, yend = y_value),
        inherit.aes = FALSE,
        color = "grey85",
        linewidth = 0.5
      ) +
      geom_line(linewidth = 2.5, alpha=0.6, na.rm = TRUE) +
      geom_point(size = 3.5, na.rm = TRUE) +
      facet_grid(model_name ~ hint_type) +
      scale_color_manual(values = metric_colors, name = "Metric") +
      scale_x_discrete(labels = x_labels) +
      scale_y_continuous(
        limits = c(0, 1),
        breaks = y_breaks,
        labels = c("0.0", ".25", ".50", ".75", "1.0")
      ) +
      labs(
        title = paste0(ds, " \u2014 ", hc),
        x = "Difficulty",
        y = "Proportion"
      ) +
      theme_minimal(base_size = 24) +
      theme(
        plot.title       = element_text(size = 36, face = "bold", hjust = 0.5),
        strip.text       = element_text(size = 28, face = "bold"),
        axis.text.x      = element_text(angle = 0, hjust = 0.5, size = 28, face = "bold"),
        axis.text.y      = element_text(size = 24, face = "bold"),
        axis.title       = element_text(size = 33, face = "bold"),
        legend.position  = "bottom",
        legend.title     = element_text(face = "bold", size = 30),
        legend.text      = element_text(size = 28, face = "bold"),
        legend.key.size  = unit(1.6, "lines"),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "white", color = NA),
        plot.background  = element_rect(fill = "white", color = NA),
        panel.spacing.y    = unit(2.0, "lines"),
        panel.spacing.x    = unit(-6.0, "lines"),
        strip.switch.pad.grid = unit(0.05, "lines"),
        strip.text.y.right = element_text(
          angle = 90,
          size = 23,
          face = "bold",
          margin = margin(l = 0, r = 0, b = 0)
        ),
        plot.margin        = margin(t = 10, r = 10, b = 10, l = 10, unit = "pt")
      ) +
      guides(color = guide_legend(nrow = 1))

    file_ds <- tolower(gsub("-", "_", gsub(" ", "_", ds)))
    file_hc <- tolower(gsub(" ", "_", hc))
    out_path <- paste0(
      "analysis/plots/verbalization_by_difficulty_",
      file_ds, "_", file_hc, "_lineplot.png"
    )
    ggsave(out_path, plot = p, width = 24, height = 12, dpi = 300)
    cat("Saved:", out_path, "\n")
  }
}
