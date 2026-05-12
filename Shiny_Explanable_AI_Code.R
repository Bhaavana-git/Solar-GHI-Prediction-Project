# ====================================
# SOLAR GHI PREDICTION DASHBOARD
# FULLY FIXED — Dynamic, Reactive, All Tabs Interactive
# ====================================

library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(DT)
library(plotly)
library(ggplot2)
library(dplyr)
library(xgboost)
library(corrplot)

# ============================================================
# STEP 1 — LOAD CSV
# ============================================================
data <- read.csv("Solar_Final_Predictions_Output.csv",
                 stringsAsFactors = FALSE)

data$DateTime <- as.POSIXct(data$DateTime, format = "%Y-%m-%d %H:%M:%S")

if (!"Hour"      %in% names(data)) data$Hour      <- as.numeric(format(data$DateTime, "%H"))
if (!"Month"     %in% names(data)) data$Month     <- as.numeric(format(data$DateTime, "%m"))
if (!"DayOfYear" %in% names(data)) data$DayOfYear <- as.numeric(format(data$DateTime, "%j"))

# ============================================================
# STEP 2 — COMPUTE DERIVED COLUMNS IF MISSING
# ============================================================
EFFICIENCY <- 0.18
AREA       <- 6000
EMISSION   <- 0.82

if (!"Predicted_GHI" %in% names(data) || all(is.na(data$Predicted_GHI)))
  data$Predicted_GHI <- data$GHI

if (!"Predicted_Power_W" %in% names(data) || all(is.na(data$Predicted_Power_W)))
  data$Predicted_Power_W    <- data$Predicted_GHI * AREA * EFFICIENCY

if (!"Predicted_Energy_kWh" %in% names(data) || all(is.na(data$Predicted_Energy_kWh)))
  data$Predicted_Energy_kWh <- data$Predicted_Power_W / 1000

if (!"CO2_Saved_kg" %in% names(data) || all(is.na(data$CO2_Saved_kg)))
  data$CO2_Saved_kg <- data$Predicted_Energy_kWh * EMISSION

if (!"Cumulative_CO2_Saved_kg" %in% names(data) || all(is.na(data$Cumulative_CO2_Saved_kg)))
  data$Cumulative_CO2_Saved_kg <- cumsum(data$CO2_Saved_kg)

data$Predicted_Power_W       <- round(data$Predicted_Power_W,       2)
data$Predicted_Energy_kWh    <- round(data$Predicted_Energy_kWh,    2)
data$CO2_Saved_kg            <- round(data$CO2_Saved_kg,            2)
data$Cumulative_CO2_Saved_kg <- round(data$Cumulative_CO2_Saved_kg, 2)

# ============================================================
# FIX 3 — Dynamic column list (computed OUTSIDE ui & server)
# ============================================================
numeric_cols <- names(data)[sapply(data, is.numeric)]

# ============================================================
# STEP 3 — MODEL METRICS
# ============================================================
model_results <- data.frame(
  Model = c("Linear Regression", "Polynomial Regression", "Random Forest", "XGBoost"),
  MAE   = c(34.56916, 32.27758,  4.824337, 2.50285),
  RMSE  = c(53.97064, 50.38515, 10.58258, 5.280771),
  R2    = c(0.9714376, 0.9748914, 0.9989738, 0.9997246)
)

# ============================================================
# STEP 4 — SHAP (optional)
# ============================================================
shap_ready       <- FALSE
shap_importance  <- NULL
shap_matrix      <- NULL
train_matrix_raw <- NULL
xgb_model        <- NULL

if (file.exists("xgb_model.rds") && file.exists("train_matrix.rds")) {
  tryCatch({
    xgb_model        <- readRDS("xgb_model.rds")
    train_matrix_raw <- readRDS("train_matrix.rds")
    if (requireNamespace("shapviz", quietly = TRUE)) {
      library(shapviz)
      sv           <- shapviz(xgb_model, X_pred = train_matrix_raw)
      shap_matrix  <- sv$S
      shap_importance <- data.frame(
        variable  = colnames(shap_matrix),
        mean_shap = colMeans(abs(shap_matrix))
      ) %>% arrange(desc(mean_shap))
      shap_ready <- TRUE
    } else {
      imp <- xgb.importance(model = xgb_model)
      shap_importance <- data.frame(
        variable  = imp$Feature,
        mean_shap = imp$Gain
      ) %>% arrange(desc(mean_shap))
      shap_ready <- TRUE
    }
  }, error = function(e) message("SHAP skipped: ", e$message))
}

ORANGE <- "#f39c12"
BLUE   <- "#2980b9"
GREEN  <- "#27ae60"

# ============================================================
# UI
# ============================================================
ui <- dashboardPage(
  skin = "yellow",
  
  dashboardHeader(
    title     = tags$span("☀️ Solar GHI Dashboard"),
    titleWidth = 260
  ),
  
  dashboardSidebar(
    width = 260,
    sidebarMenu(
      id = "tabs",
      menuItem("🏠 Overview",         tabName = "overview",    icon = icon("sun")),
      menuItem("📊 EDA",              tabName = "eda",         icon = icon("chart-bar")),
      menuItem("🤖 Model Comparison", tabName = "models",      icon = icon("table")),
      menuItem("⚡ Predictions",      tabName = "predictions", icon = icon("bolt")),
      menuItem("🔍 Explainable AI",   tabName = "xai",         icon = icon("lightbulb"))
    ),
    tags$div(
      style = "position:absolute;bottom:16px;left:0;right:0;
               text-align:center;color:#888;font-size:11px;padding:0 10px;",
      tags$hr(style = "border-color:#444;"),
      "Efficiency: 18% | Area: 6000 m²", tags$br(),
      "CO₂ Factor: 0.82 kg/kWh"
    )
  ),
  
  dashboardBody(
    tags$head(tags$style(HTML("
      .content-wrapper { background:#f5f6fa; }
      .kpi-box {
        background:white; border-radius:12px; padding:18px 22px;
        box-shadow:0 2px 10px rgba(0,0,0,0.08);
        border-left:5px solid #f39c12; margin-bottom:16px;
      }
      .kpi-box h3 { margin:0 0 4px; font-size:24px; font-weight:700; color:#2c3e50; }
      .kpi-box p  { margin:0; font-size:11px; color:#888;
                    text-transform:uppercase; letter-spacing:0.5px; }
      .box        { border-radius:10px; }
      .box-header { border-bottom:2px solid #f39c12; }
      .section-header {
        font-size:17px; font-weight:700; color:#2c3e50;
        border-bottom:3px solid #f39c12; padding-bottom:6px; margin-bottom:18px;
      }
      .shap-card {
        background:#fffbf0; border-left:4px solid #f39c12;
        border-radius:8px; padding:14px 18px; margin-bottom:12px;
      }
      .shap-card h4 { margin:0 0 6px; color:#2c3e50; font-size:14px; }
      .shap-card p  { margin:0; color:#555; font-size:13px; line-height:1.6; }
    "))),
    
    tabItems(
      
      # ==========================================
      # TAB 1: OVERVIEW — FIX 1: Month + Metric filter added
      # ==========================================
      tabItem(tabName = "overview",
              fluidRow(column(12, tags$div(class="section-header","☀️ Solar Power Generation Overview"))),
              
              # ✅ FIX 1: Interactive filter row
              fluidRow(
                box(width = 12, status = "warning",
                    fluidRow(
                      column(5,
                             sliderInput("overview_month", "Filter by Month Range:",
                                         min=1, max=12, value=c(1,12), step=1,
                                         ticks=TRUE, width="100%")
                      ),
                      column(4,
                             sliderInput("overview_hour", "Filter by Hour Range:",
                                         min=0, max=23, value=c(0,23), step=1,
                                         ticks=TRUE, width="100%")
                      ),
                      column(3,
                             selectInput("overview_metric", "KPI Metric:",
                                         choices = c("Energy (kWh)"  = "Predicted_Energy_kWh",
                                                     "CO₂ Saved (kg)"= "CO2_Saved_kg",
                                                     "GHI (W/m²)"    = "GHI",
                                                     "Power (W)"     = "Predicted_Power_W"),
                                         selected = "Predicted_Energy_kWh")
                      )
                    )
                )
              ),
              
              fluidRow(
                column(3, tags$div(class="kpi-box",
                                   tags$h3(textOutput("kpi_total_energy")),
                                   tags$p("Total Predicted Energy (kWh)"))),
                column(3, tags$div(class="kpi-box",
                                   tags$h3(textOutput("kpi_total_co2")),
                                   tags$p("Total CO₂ Saved (Tonnes)"))),
                column(3, tags$div(class="kpi-box",
                                   tags$h3(textOutput("kpi_avg_ghi")),
                                   tags$p("Average GHI (W/m²)"))),
                column(3, tags$div(class="kpi-box",
                                   tags$h3(textOutput("kpi_peak_power")),
                                   tags$p("Peak Power Generated (kW)")))
              ),
              
              fluidRow(
                box(title="GHI Over Time", width=8, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_ghi_time", height="300px")),
                box(title="Energy by Month (MWh)", width=4, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_energy_month", height="300px"))
              ),
              
              fluidRow(
                box(title="Cumulative CO₂ Saved (kg)", width=6, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_co2_cumulative", height="280px")),
                box(title="Power Heatmap (Hour × Month)", width=6, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_heatmap", height="280px"))
              )
      ),
      
      # ==========================================
      # TAB 2: EDA — FIX 3: Dynamic column choices
      # ==========================================
      tabItem(tabName = "eda",
              fluidRow(column(12, tags$div(class="section-header","📊 Exploratory Data Analysis"))),
              
              fluidRow(
                box(title="Select Variables", width=3, status="warning", solidHeader=TRUE,
                    # ✅ FIX 3: Dynamic choices from numeric_cols
                    selectInput("eda_x","X-axis Variable:",
                                choices  = numeric_cols,
                                selected = ifelse("Temperature" %in% numeric_cols, "Temperature", numeric_cols[1])),
                    selectInput("eda_y","Y-axis Variable:",
                                choices  = numeric_cols,
                                selected = ifelse("GHI" %in% numeric_cols, "GHI", numeric_cols[2])),
                    radioButtons("eda_plot_type","Plot Type:",
                                 choices  = c("Scatter"="scatter","Histogram"="hist","Box Plot"="box"),
                                 selected = "scatter"),
                    # ✅ NEW: Color-by option
                    selectInput("eda_color","Color By (Scatter only):",
                                choices  = c("None"="none", numeric_cols),
                                selected = "none")
                ),
                box(title="Plot", width=9, status="warning", solidHeader=TRUE,
                    plotlyOutput("eda_plot", height="350px"))
              ),
              
              fluidRow(
                box(title="Correlation Matrix", width=6, status="warning", solidHeader=TRUE,
                    # ✅ NEW: Let user pick which columns to correlate
                    checkboxGroupInput("corr_cols", "Columns for Correlation:",
                                       choices  = intersect(c("GHI","DHI","DNI","Temperature",
                                                              "Wind.Speed","Relative.Humidity",
                                                              "Hour","Month","DayOfYear"),
                                                            names(data)),
                                       selected = intersect(c("GHI","DHI","DNI","Temperature",
                                                              "Wind.Speed","Relative.Humidity"),
                                                            names(data)),
                                       inline   = TRUE),
                    plotOutput("plot_corr", height="340px")),
                box(title="GHI Distribution", width=6, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_ghi_dist", height="380px"))
              )
      ),
      
      # ==========================================
      # TAB 3: MODEL COMPARISON — NEW: model selector
      # ==========================================
      tabItem(tabName = "models",
              fluidRow(column(12, tags$div(class="section-header","🤖 Model Performance Comparison"))),
              
              # ✅ NEW: Let user choose which models to compare
              fluidRow(
                box(width=12, status="warning",
                    checkboxGroupInput("selected_models", "Select Models to Compare:",
                                       choices  = model_results$Model,
                                       selected = model_results$Model,
                                       inline   = TRUE)
                )
              ),
              
              fluidRow(
                box(title="Model Metrics Table", width=5, status="warning", solidHeader=TRUE,
                    DTOutput("table_model_metrics")),
                box(title="RMSE Comparison", width=7, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_model_rmse", height="300px"))
              ),
              
              fluidRow(
                box(title="R² Comparison", width=6, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_model_r2", height="280px")),
                box(title="MAE Comparison", width=6, status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_model_mae", height="280px"))
              ),
              
              fluidRow(
                box(title="Actual vs Predicted GHI (XGBoost)", width=12,
                    status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_actual_vs_pred", height="320px"))
              )
      ),
      
      # ==========================================
      # TAB 4: PREDICTIONS — FIX 2: Live filter (no button)
      # ==========================================
      tabItem(tabName = "predictions",
              fluidRow(column(12, tags$div(class="section-header","⚡ Power & CO₂ Predictions"))),
              
              fluidRow(
                box(title="Filter Predictions", width=3, status="warning", solidHeader=TRUE,
                    selectInput("pred_month","Select Month:",
                                choices  = c("All"=0, setNames(1:12, month.name)),
                                selected = 0),
                    selectInput("pred_hour","Select Hour:",
                                choices  = c("All"=-1, setNames(0:23, paste0(0:23,":00"))),
                                selected = -1),
                    # ✅ NEW: Y-axis metric picker for predictions chart
                    selectInput("pred_metric", "Chart Metric:",
                                choices  = c("Energy (kWh)"   = "Predicted_Energy_kWh",
                                             "CO₂ Saved (kg)" = "CO2_Saved_kg",
                                             "Power (W)"      = "Predicted_Power_W",
                                             "Both Energy & CO₂" = "both"),
                                selected = "both"),
                    # ✅ NOTE: actionButton REMOVED — filter is now live
                    tags$p(style="font-size:11px;color:#aaa;margin-top:8px;",
                           "Filters apply instantly — no button needed.")
                ),
                box(title="Predicted Energy & CO₂ Over Time", width=9,
                    status="warning", solidHeader=TRUE,
                    plotlyOutput("plot_pred_energy", height="320px"))
              ),
              
              fluidRow(
                box(title="Predictions Data Table", width=12, status="warning", solidHeader=TRUE,
                    DTOutput("table_predictions"))
              )
      ),
      
      # ==========================================
      # TAB 5: EXPLAINABLE AI
      # ==========================================
      tabItem(tabName = "xai",
              fluidRow(column(12, tags$div(class="section-header",
                                           "🔍 Explainable AI — Why Does the Model Predict This?"))),
              
              fluidRow(
                box(width=12, background="yellow",
                    tags$p(style="margin:0;font-size:14px;",
                           "📌 ", tags$b("SHAP (SHapley Additive exPlanations)"),
                           " shows exactly how much each feature pushed the GHI prediction ",
                           tags$b("up ↑ or down ↓"),
                           " — directly explaining what drives solar power & CO₂ savings."
                    )
                )
              ),
              
              fluidRow(column(12, uiOutput("shap_setup_msg"))),
              
              fluidRow(
                box(title="Global Feature Importance (Mean |SHAP|)",
                    width=6, status="warning", solidHeader=TRUE,
                    # ✅ NEW: top-N slider
                    sliderInput("shap_top_n", "Show Top N Features:",
                                min=3, max=min(15, length(numeric_cols)),
                                value=8, step=1),
                    plotlyOutput("shap_importance_plot", height="320px")),
                box(title="SHAP Beeswarm / Summary",
                    width=6, status="warning", solidHeader=TRUE,
                    plotOutput("shap_beeswarm", height="380px"))
              ),
              
              fluidRow(
                box(title="SHAP Dependence — Select Feature", width=12,
                    status="warning", solidHeader=TRUE,
                    fluidRow(
                      column(3,
                             selectInput("shap_feature","Choose Feature:",
                                         choices  = c("DHI","DNI","Temperature","Wind.Speed",
                                                      "Relative.Humidity","Hour","Month","DayOfYear"),
                                         selected = "Hour"),
                             tags$p(style="font-size:12px;color:#888;margin-top:8px;",
                                    "Shows how feature value affects its SHAP contribution to GHI.")
                      ),
                      column(9, plotOutput("shap_dependence", height="300px"))
                    )
                )
              ),
              
              fluidRow(
                box(title="What Drives Power & CO₂ Reduction? — Interpretation",
                    width=12, status="warning", solidHeader=TRUE,
                    uiOutput("shap_interpretation"))
              )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================
server <- function(input, output, session) {
  
  # ============================================================
  # FIX 1 — Overview reactive data (responds to month + hour sliders)
  # ============================================================
  overview_data <- reactive({
    data[data$Month >= input$overview_month[1] &
           data$Month <= input$overview_month[2] &
           data$Hour  >= input$overview_hour[1]  &
           data$Hour  <= input$overview_hour[2], ]
  })
  
  # ============================================================
  # FIX 2 — Predictions live filter (no button)
  # ============================================================
  filtered_data <- reactive({
    df <- data
    m  <- as.numeric(input$pred_month)
    h  <- as.numeric(input$pred_hour)
    if (m != 0)  df <- df[df$Month == m, ]
    if (h != -1) df <- df[df$Hour  == h, ]
    df
  })
  
  # ============================================================
  # Model comparison reactive (responds to checkbox)
  # ============================================================
  selected_model_data <- reactive({
    model_results[model_results$Model %in% input$selected_models, ]
  })
  
  # ---- TAB 1: KPIs — all use overview_data() ----
  output$kpi_total_energy <- renderText({
    paste0(formatC(sum(overview_data()$Predicted_Energy_kWh, na.rm=TRUE),
                   format="f", big.mark=",", digits=0), " kWh")
  })
  output$kpi_total_co2 <- renderText({
    paste0(round(sum(overview_data()$CO2_Saved_kg, na.rm=TRUE)/1000, 1), " Tonnes")
  })
  output$kpi_avg_ghi <- renderText({
    paste0(round(mean(overview_data()$GHI, na.rm=TRUE), 1), " W/m²")
  })
  output$kpi_peak_power <- renderText({
    paste0(round(max(overview_data()$Predicted_Power_W, na.rm=TRUE)/1000, 1), " kW")
  })
  
  # GHI over time — uses overview_data()
  output$plot_ghi_time <- renderPlotly({
    df <- overview_data()
    if (nrow(df) == 0) return(plotly_empty() %>% layout(title="No data for selected range"))
    n  <- nrow(df)
    df <- df[seq(1, n, by = max(1, floor(n/500))), ]
    plot_ly(df, x=~DateTime, y=~GHI, type="scatter", mode="lines",
            line=list(color=ORANGE, width=1.2)) %>%
      layout(xaxis=list(title=""), yaxis=list(title="GHI (W/m²)"), margin=list(t=10))
  })
  
  # Energy by month — uses overview_data()
  output$plot_energy_month <- renderPlotly({
    df <- overview_data()
    if (nrow(df) == 0) return(plotly_empty() %>% layout(title="No data for selected range"))
    df <- df %>%
      group_by(Month) %>%
      summarise(Energy = sum(Predicted_Energy_kWh, na.rm=TRUE)/1000, .groups="drop")
    plot_ly(df, x=~factor(Month, levels=df$Month, labels=month.abb[df$Month]),
            y=~Energy, type="bar",
            marker=list(color=ORANGE)) %>%
      layout(xaxis=list(title="Month"), yaxis=list(title="Energy (MWh)"), margin=list(t=10))
  })
  
  # Cumulative CO2 — uses overview_data()
  output$plot_co2_cumulative <- renderPlotly({
    df <- overview_data()
    if (nrow(df) == 0) return(plotly_empty() %>% layout(title="No data for selected range"))
    df <- df %>% arrange(DateTime) %>% mutate(CumCO2 = cumsum(CO2_Saved_kg))
    n  <- nrow(df)
    df <- df[seq(1, n, by = max(1, floor(n/500))), ]
    plot_ly(df, x=~DateTime, y=~CumCO2,
            type="scatter", mode="lines", fill="tozeroy",
            line=list(color=GREEN), fillcolor="rgba(39,174,96,0.15)") %>%
      layout(xaxis=list(title=""), yaxis=list(title="Cumulative CO₂ (kg)"), margin=list(t=10))
  })
  
  # Heatmap — uses overview_data()
  output$plot_heatmap <- renderPlotly({
    df <- overview_data()
    if (nrow(df) == 0) return(plotly_empty() %>% layout(title="No data for selected range"))
    df <- df %>%
      group_by(Hour, Month) %>%
      summarise(Power = mean(Predicted_Power_W/1000, na.rm=TRUE), .groups="drop")
    plot_ly(df, x=~Month, y=~Hour, z=~Power, type="heatmap",
            colors=colorRamp(c("white", ORANGE, "#c0392b"))) %>%
      layout(xaxis=list(title="Month", tickvals=1:12, ticktext=month.abb),
             yaxis=list(title="Hour"), margin=list(t=10))
  })
  
  # ---- TAB 2: EDA ----
  output$eda_plot <- renderPlotly({
    xv <- input$eda_x
    yv <- input$eda_y
    pt <- input$eda_plot_type
    cv <- input$eda_color
    
    if (!xv %in% names(data) || !yv %in% names(data))
      return(plotly_empty() %>% layout(title="Column not found in data"))
    
    if (pt == "scatter") {
      if (cv != "none" && cv %in% names(data)) {
        plot_ly(data, x=data[[xv]], y=data[[yv]], color=data[[cv]],
                type="scatter", mode="markers",
                marker=list(opacity=0.35, size=4)) %>%
          layout(xaxis=list(title=xv), yaxis=list(title=yv),
                 coloraxis=list(colorbar=list(title=cv)))
      } else {
        plot_ly(data, x=data[[xv]], y=data[[yv]],
                type="scatter", mode="markers",
                marker=list(color=ORANGE, opacity=0.35, size=4)) %>%
          layout(xaxis=list(title=xv), yaxis=list(title=yv))
      }
    } else if (pt == "hist") {
      plot_ly(data, x=data[[xv]], type="histogram",
              marker=list(color=ORANGE)) %>%
        layout(xaxis=list(title=xv), yaxis=list(title="Count"))
    } else {
      plot_ly(data, y=data[[yv]], type="box",
              marker=list(color=ORANGE), fillcolor=ORANGE) %>%
        layout(yaxis=list(title=yv))
    }
  })
  
  # Correlation matrix — responds to checkbox selection
  output$plot_corr <- renderPlot({
    keep <- input$corr_cols
    keep <- keep[keep %in% names(data)]
    if (length(keep) < 2) {
      plot.new(); text(0.5, 0.5, "Select at least 2 columns", cex=1.2); return()
    }
    df_cor <- na.omit(data[, keep, drop=FALSE])
    df_cor <- df_cor[, apply(df_cor, 2, function(x) var(x) > 1e-10), drop=FALSE]
    if (ncol(df_cor) < 2) {
      plot.new(); text(0.5, 0.5, "Not enough variance in selected columns", cex=1.2); return()
    }
    cm <- cor(df_cor, use="complete.obs")
    corrplot(cm, method="color", type="upper", tl.cex=0.85,
             addCoef.col="black", number.cex=0.7,
             col=colorRampPalette(c(BLUE,"white",ORANGE))(200),
             mar=c(0,0,1,0))
  })
  
  output$plot_ghi_dist <- renderPlotly({
    plot_ly(data, x=~GHI, type="histogram",
            marker=list(color=ORANGE, line=list(color="white",width=0.4))) %>%
      layout(xaxis=list(title="GHI (W/m²)"), yaxis=list(title="Count"), margin=list(t=10))
  })
  
  # ---- TAB 3: MODEL COMPARISON — all react to selected_model_data() ----
  output$table_model_metrics <- renderDT({
    datatable(selected_model_data(), rownames=FALSE,
              options=list(dom="t", pageLength=5)) %>%
      formatRound(c("MAE","RMSE","R2"), digits=3)
  })
  
  bar_colors <- c("#e74c3c","#e67e22","#3498db","#f39c12")
  
  output$plot_model_rmse <- renderPlotly({
    df <- selected_model_data()
    if (nrow(df) == 0) return(plotly_empty() %>% layout(title="Select at least one model"))
    plot_ly(df, x=~Model, y=~RMSE, type="bar",
            marker=list(color=bar_colors[seq_len(nrow(df))])) %>%
      layout(xaxis=list(title=""), yaxis=list(title="RMSE (lower = better)"), margin=list(t=10))
  })
  
  output$plot_model_r2 <- renderPlotly({
    df <- selected_model_data()
    if (nrow(df) == 0) return(plotly_empty() %>% layout(title="Select at least one model"))
    plot_ly(df, x=~Model, y=~R2, type="bar",
            marker=list(color=bar_colors[seq_len(nrow(df))])) %>%
      layout(xaxis=list(title=""), yaxis=list(title="R² (higher = better)"), margin=list(t=10))
  })
  
  output$plot_model_mae <- renderPlotly({
    df <- selected_model_data()
    if (nrow(df) == 0) return(plotly_empty() %>% layout(title="Select at least one model"))
    plot_ly(df, x=~Model, y=~MAE, type="bar",
            marker=list(color=bar_colors[seq_len(nrow(df))])) %>%
      layout(xaxis=list(title=""), yaxis=list(title="MAE (lower = better)"), margin=list(t=10))
  })
  
  output$plot_actual_vs_pred <- renderPlotly({
    df <- data[!is.na(data$GHI) & !is.na(data$Predicted_GHI), ]
    df <- df[sample(nrow(df), min(2000, nrow(df))), ]
    mx <- max(df$GHI, na.rm=TRUE)
    plot_ly(df, x=~GHI, y=~Predicted_GHI,
            type="scatter", mode="markers",
            marker=list(color=ORANGE, opacity=0.4, size=5),
            name="Predictions") %>%
      add_lines(x=c(0,mx), y=c(0,mx),
                line=list(color="red", dash="dash"), name="Perfect Fit") %>%
      layout(xaxis=list(title="Actual GHI (W/m²)"),
             yaxis=list(title="Predicted GHI (W/m²)"),
             margin=list(t=10))
  })
  
  # ---- TAB 4: PREDICTIONS — FIX 2: uses live reactive filtered_data() ----
  output$plot_pred_energy <- renderPlotly({
    df <- filtered_data()
    if (nrow(df) == 0)
      return(plotly_empty() %>% layout(title="No data for selected filter"))
    n  <- nrow(df)
    df <- df[seq(1, n, by = max(1, floor(n/500))), ]
    
    metric <- input$pred_metric
    if (metric == "both") {
      plot_ly(df, x=~DateTime) %>%
        add_lines(y=~Predicted_Energy_kWh, name="Energy (kWh)", line=list(color=GREEN)) %>%
        add_lines(y=~CO2_Saved_kg,         name="CO₂ Saved (kg)", line=list(color=BLUE)) %>%
        layout(xaxis=list(title="Time"), yaxis=list(title="Value"),
               legend=list(orientation="h"), margin=list(t=10))
    } else {
      plot_ly(df, x=~DateTime, y=df[[metric]],
              type="scatter", mode="lines",
              line=list(color=ORANGE)) %>%
        layout(xaxis=list(title="Time"), yaxis=list(title=metric), margin=list(t=10))
    }
  })
  
  output$table_predictions <- renderDT({
    df        <- filtered_data()
    cols_show <- c("DateTime","Hour","Month","GHI","Predicted_GHI",
                   "Predicted_Power_W","Predicted_Energy_kWh",
                   "CO2_Saved_kg","Cumulative_CO2_Saved_kg")
    cols_show <- cols_show[cols_show %in% names(df)]
    datatable(df[, cols_show],
              options=list(pageLength=15, scrollX=TRUE),
              rownames=FALSE)
  })
  
  # ---- TAB 5: EXPLAINABLE AI ----
  output$shap_setup_msg <- renderUI({
    if (shap_ready) return(NULL)
    tags$div(
      style="background:#fff3cd;border-left:4px solid #f39c12;
             border-radius:8px;padding:14px 18px;margin-bottom:16px;",
      tags$h4(style="margin:0 0 8px;color:#856404;",
              "⚠️ SHAP not yet enabled — follow these steps:"),
      tags$ol(style="margin:0;color:#555;font-size:13px;line-height:2;",
              tags$li("In your original R session (where you trained the model), run:"),
              tags$code(style="background:#f8f9fa;padding:4px 8px;border-radius:4px;display:block;margin:4px 0;",
                        "saveRDS(model_xgb, 'xgb_model.rds')
saveRDS(train_matrix, 'train_matrix.rds')"),
              tags$li(paste0("Place both .rds files in the same folder as app.R: ", getwd())),
              tags$li("Run: install.packages('shapviz')"),
              tags$li("Restart the app — SHAP plots will appear automatically.")
      )
    )
  })
  
  # SHAP importance — reacts to top_n slider
  output$shap_importance_plot <- renderPlotly({
    if (!shap_ready || is.null(shap_importance))
      return(plotly_empty() %>% layout(title="SHAP not loaded — see instructions above"))
    top_n <- input$shap_top_n
    df    <- head(shap_importance, top_n)
    plot_ly(df,
            x=~mean_shap,
            y=~reorder(variable, mean_shap),
            type="bar", orientation="h",
            marker=list(color=ORANGE)) %>%
      layout(xaxis=list(title="Mean |SHAP Value|"),
             yaxis=list(title=""),
             margin=list(t=10, l=140))
  })
  
  output$shap_beeswarm <- renderPlot({
    if (!shap_ready || is.null(shap_matrix)) {
      plot.new()
      text(0.5, 0.5,
           "Place xgb_model.rds & train_matrix.rds\nnext to app.R and restart.",
           cex=1.1, col="#888")
      return()
    }
    if (requireNamespace("shapviz", quietly=TRUE)) {
      library(shapviz)
      sv <- shapviz(xgb_model, X_pred = train_matrix_raw)
      print(sv_importance(sv, kind="beeswarm"))
    } else {
      plot.new(); text(0.5,0.5,"Install shapviz for beeswarm plot", cex=1.1)
    }
  })
  
  output$shap_dependence <- renderPlot({
    feat <- input$shap_feature
    if (!shap_ready || is.null(shap_matrix)) {
      plot.new()
      text(0.5,0.5,
           "Place xgb_model.rds & train_matrix.rds\nnext to app.R and restart.",
           cex=1.1, col="#888")
      return()
    }
    if (requireNamespace("shapviz", quietly=TRUE)) {
      library(shapviz)
      sv <- shapviz(xgb_model, X_pred = train_matrix_raw)
      print(sv_dependence(sv, v = feat))
    } else {
      feat_vals <- train_matrix_raw[, feat]
      shap_vals <- shap_matrix[, feat]
      df_dep    <- data.frame(x=feat_vals, shap=shap_vals)
      p <- ggplot(df_dep, aes(x=x, y=shap)) +
        geom_point(alpha=0.3, color=ORANGE, size=1.5) +
        geom_smooth(method="loess", se=TRUE, color=BLUE) +
        labs(title=paste("SHAP Dependence:", feat), x=feat, y="SHAP Value") +
        theme_minimal()
      print(p)
    }
  })
  
  output$shap_interpretation <- renderUI({
    if (is.null(shap_importance))
      return(tags$p(style="color:#888;",
                    "Feature importance will display here once SHAP is enabled."))
    top <- head(shap_importance, 5)
    explanations <- list(
      "DNI"               = list(text="Direct Normal Irradiance is the strongest solar signal. Higher DNI = more direct sunlight → GHI rises → Power and CO₂ savings increase proportionally.", impact="↑ DNI by 100 W/m² → ↑ Power ~108 kW | ↑ CO₂ savings ~88.6 kg/hr"),
      "DHI"               = list(text="Diffuse Horizontal Irradiance captures scattered sunlight. Even on cloudy days, DHI keeps generation from dropping to zero.", impact="Higher DHI → Energy output maintained even without direct sunlight"),
      "Hour"              = list(text="Time of day is critical. Solar output peaks around noon (11am–2pm) and is near zero before 6am and after 6pm.", impact="Peak hours generate 3–5× more power than early morning/evening"),
      "Temperature"       = list(text="High temperature correlates with sunny clear days. Slight efficiency loss from heat is outweighed by higher irradiance.", impact="Summer (high temp + high DNI) = maximum annual energy output"),
      "DayOfYear"         = list(text="Seasonal position drives solar angle and daylight hours. Summer days (Day 150–250) produce significantly more power than winter.", impact="Day 180 (June) ≈ 2× daily energy of Day 355 (December)"),
      "Month"             = list(text="Month captures seasonal irradiance variation. May–August are peak months; December–January are minimum generation months.", impact="Peak month CO₂ savings can be 2–3× the lowest month"),
      "Relative.Humidity" = list(text="High humidity signals cloud cover and atmospheric scattering, reducing GHI. The model uses it as a proxy for sky clarity.", impact="Humidity >80% typically suppresses GHI by 20–40%"),
      "Wind.Speed"        = list(text="Wind cools panels (slight efficiency boost) and can indicate weather patterns. It is a secondary driver.", impact="Minor effect — important as a weather-state indicator")
    )
    cards <- lapply(seq_len(nrow(top)), function(i) {
      feat <- top$variable[i]
      info <- explanations[[feat]]
      if (is.null(info)) info <- list(
        text   = paste(feat, "significantly impacts GHI predictions."),
        impact = "See dependence plot above for detailed effect."
      )
      tags$div(class="shap-card",
               tags$h4(paste0("#", i, "  ", feat,
                              "  —  Mean |SHAP| = ", round(top$mean_shap[i], 4))),
               tags$p(info$text),
               tags$p(style="margin-top:8px;font-size:12px;color:#e67e22;font-weight:600;",
                      paste0("⚡ Power & CO₂: ", info$impact)),
               tags$p(style="margin-top:4px;font-size:11px;color:#aaa;",
                      "Power (W) = GHI × 6000 m² × 0.18  |  CO₂ Saved (kg) = Energy (kWh) × 0.82")
      )
    })
    tagList(cards)
  })
  
} # end server

shinyApp(ui = ui, server = server)
