"""
Streamlit Dashboard for Bank Check Prediction System

Simple dashboard for making predictions and viewing results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Direct imports to avoid package complexity
from src.models.prediction_model import CheckPredictionModel
from src.models.model_manager import ModelManager
from src.data_processing.dataset_builder import DatasetBuilder

# Configure page
st.set_page_config(
    page_title="Bank Check Prediction Dashboard",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

def load_prediction_model():
    """Load the prediction model."""
    try:
        model = CheckPredictionModel()
        
        # Try to load pre-trained model
        model_path = Path("data/models/prediction_model.json")
        if model_path.exists():
            model.load_model(str(model_path))
            return model
        else:
            st.error("No pre-trained model found. Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def load_dataset():
    """Load the processed dataset."""
    try:
        dataset_path = Path("data/processed/dataset_final.csv")
        
        if dataset_path.exists():
            return pd.read_csv(dataset_path)
        else:
            st.warning("Dataset not found. Please run the data processing pipeline first.")
            return None
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def main():
    """Main dashboard application."""
    
    # Header
    st.title("🏦 Bank Check Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "🔮 Predictions",
            "📊 Model Performance", 
            "📈 Data Analytics",
            "⚙️ Model Management"
        ]
    )
    
    # Load model and dataset if not already loaded
    if st.session_state.prediction_model is None:
        with st.spinner("Loading prediction model..."):
            st.session_state.prediction_model = load_prediction_model()
    
    if st.session_state.dataset is None:
        with st.spinner("Loading dataset..."):
            st.session_state.dataset = load_dataset()
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Predictions":
        show_predictions_page()
    elif page == "📊 Model Performance":
        show_performance_page()
    elif page == "📈 Data Analytics":
        show_analytics_page()
    elif page == "⚙️ Model Management":
        show_management_page()

def show_home_page():
    """Display the home page."""
    
    st.header("Welcome to the Bank Check Prediction System")
    
    # Overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model Status",
            value="Ready" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "Not Ready",
            delta="Trained" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "Needs Training"
        )
    
    with col2:
        dataset_size = len(st.session_state.dataset) if st.session_state.dataset is not None else 0
        st.metric(
            label="Dataset Size",
            value=f"{dataset_size:,}",
            delta="Records"
        )
    
    with col3:
        st.metric(
            label="Version",
            value="1.0.0",
            delta="Production"
        )
    
    with col4:
        st.metric(
            label="Features",
            value="15",
            delta="ML Features"
        )
    
    st.markdown("---")
    
    # System Overview
    st.subheader("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Objectives
        - **Predict number of checks** a client will issue
        - **Predict maximum authorized amount** per check
        - **Analyze client behavior** patterns
        - **Support decision making** for check allocation
        """)
        
        st.markdown("""
        ### Features
        - **User-selectable models** with 3 ML algorithms
        - **Real-time predictions** for banking applications
        - **Interactive dashboard** for analysis
        - **Model performance monitoring**
        """)
    
    with col2:
        if st.session_state.prediction_model and st.session_state.prediction_model.is_trained:
            metrics = st.session_state.prediction_model.metrics
            
            st.markdown("### Model Performance")
            
            # Create metrics visualization
            fig = go.Figure()
            
            models = ['Number of Checks', 'Maximum Amount']
            r2_scores = [
                metrics.get('nbr_cheques', {}).get('r2', 0),
                metrics.get('montant_max', {}).get('r2', 0)
            ]
            
            fig.add_trace(go.Bar(
                x=models,
                y=r2_scores,
                name='R² Score',
                marker_color=['#FF6B6B', '#4ECDC4']
            ))
            
            fig.update_layout(
                title="Model R² Scores",
                yaxis_title="R² Score",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model not loaded. Please check the model management page.")

def show_predictions_page():
    """Display the predictions page."""
    
    st.header("Client Predictions")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("Prediction model is not available. Please check the model management page.")
        return
    
    # Single client prediction
    st.subheader("Single Client Prediction")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Client Information")
            client_id = st.text_input("Client ID", value="client_test_001")
            marche = st.selectbox("Market", ["Particuliers", "PME", "TPE", "GEI", "TRE", "PRO"])
            csp = st.text_input("CSP", value="Cadre")
            segment = st.text_input("Segment", value="Segment_A")
            secteur = st.text_input("Activity Sector", value="Services")
            
        with col2:
            st.markdown("### Financial Information")
            revenu = st.number_input("Estimated Revenue", min_value=0.0, value=50000.0)
            nbr_2024 = st.number_input("Number of Checks 2024", min_value=0, value=5)
            montant_2024 = st.number_input("Max Amount 2024", min_value=0.0, value=30000.0)
            ecart_nbr = st.number_input("Check Number Difference", value=2)
            ecart_montant = st.number_input("Amount Difference", value=5000.0)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Behavioral Information")
            demande_derogation = st.checkbox("Has Requested Derogation")
            mobile_banking = st.checkbox("Uses Mobile Banking")
            ratio_cheques = st.slider("Check Payment Ratio", 0.0, 1.0, 0.3)
            
        with col4:
            st.markdown("### Payment Information")
            nb_methodes = st.number_input("Number of Payment Methods", min_value=0, value=3)
            montant_moyen_cheque = st.number_input("Average Check Amount", min_value=0.0, value=1500.0)
            montant_moyen_alt = st.number_input("Average Alternative Amount", min_value=0.0, value=800.0)
        
        submitted = st.form_submit_button("Predict", use_container_width=True)
        
        if submitted:
            # Prepare client data
            client_data = {
                'CLI': client_id,
                'CLIENT_MARCHE': marche,
                'CSP': csp,
                'Segment_NMR': segment,
                'CLT_SECTEUR_ACTIVITE_LIB': secteur,
                'Revenu_Estime': revenu,
                'Nbr_Cheques_2024': nbr_2024,
                'Montant_Max_2024': montant_2024,
                'Ecart_Nbr_Cheques_2024_2025': ecart_nbr,
                'Ecart_Montant_Max_2024_2025': ecart_montant,
                'A_Demande_Derogation': int(demande_derogation),
                'Ratio_Cheques_Paiements': ratio_cheques,
                'Utilise_Mobile_Banking': int(mobile_banking),
                'Nombre_Methodes_Paiement': nb_methodes,
                'Montant_Moyen_Cheque': montant_moyen_cheque,
                'Montant_Moyen_Alternative': montant_moyen_alt
            }
            
            # Make prediction
            try:
                result = st.session_state.prediction_model.predict(client_data)
                
                # Display results
                st.success("Prediction completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Number of Checks",
                        value=result['predicted_nbr_cheques'],
                        delta=f"vs {nbr_2024} in 2024"
                    )
                
                with col2:
                    st.metric(
                        label="Predicted Maximum Amount",
                        value=f"€{result['predicted_montant_max']:,.2f}",
                        delta=f"vs €{montant_2024:,.2f} in 2024"
                    )
                
                with col3:
                    confidence = result['model_confidence']
                    avg_confidence = (confidence['nbr_cheques_r2'] + confidence['montant_max_r2']) / 2
                    st.metric(
                        label="Model Confidence",
                        value=f"{avg_confidence:.1%}",
                        delta="Average R² Score"
                    )
                
                # Detailed results
                with st.expander("Detailed Results"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

def show_performance_page():
    """Display model performance page."""
    
    st.header("Model Performance Analysis")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("Model not available. Please check the model management page.")
        return
    
    metrics = st.session_state.prediction_model.metrics
    
    # Model selection info
    if st.session_state.prediction_model and st.session_state.prediction_model.is_trained:
        model_info = st.session_state.prediction_model.get_model_info()
        selected_model = model_info.get('model_type', 'unknown')
        
        model_names = {
            'linear': 'Linear Regression',
            'gradient_boost': 'Gradient Boosting',
            'neural_network': 'Neural Network'
        }
        
        st.info(f"**Current Model**: {model_names.get(selected_model, selected_model)}")
    
    # Performance metrics overview
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Number of Checks Model")
        nbr_metrics = metrics.get('nbr_cheques', {})
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("R² Score", f"{nbr_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{nbr_metrics.get('mae', 0):.4f}")
        with metric_col2:
            st.metric("MSE", f"{nbr_metrics.get('mse', 0):.4f}")
            st.metric("RMSE", f"{nbr_metrics.get('rmse', 0):.4f}")
    
    with col2:
        st.markdown("### Maximum Amount Model")
        montant_metrics = metrics.get('montant_max', {})
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("R² Score", f"{montant_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{montant_metrics.get('mae', 0):,.2f}")
        with metric_col2:
            st.metric("MSE", f"{montant_metrics.get('mse', 0):,.0f}")
            st.metric("RMSE", f"{montant_metrics.get('rmse', 0):,.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    importance = st.session_state.prediction_model.get_feature_importance()
    if importance:
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (Based on Model Weights)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def show_analytics_page():
    """Display data analytics page."""
    
    st.header("Data Analytics & Insights")
    
    if st.session_state.dataset is None:
        st.error("Dataset not available. Please check the data processing pipeline.")
        return
    
    df = st.session_state.dataset
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clients", len(df))
    with col2:
        avg_checks = df['Target_Nbr_Cheques_Futur'].mean()
        st.metric("Avg Checks", f"{avg_checks:.1f}")
    with col3:
        avg_amount = df['Target_Montant_Max_Futur'].mean()
        st.metric("Avg Max Amount", f"€{avg_amount:,.0f}")
    with col4:
        derogation_rate = df['A_Demande_Derogation'].mean() * 100
        st.metric("Derogation Rate", f"{derogation_rate:.1f}%")
    
    # Market distribution
    st.subheader("Market Distribution")
    
    market_counts = df['CLIENT_MARCHE'].value_counts()
    fig = px.pie(values=market_counts.values, names=market_counts.index, title="Client Distribution by Market")
    st.plotly_chart(fig, use_container_width=True)
    
    # Target distribution
    st.subheader("Target Variables Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='Target_Nbr_Cheques_Futur',
            title="Distribution of Number of Checks"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='Target_Montant_Max_Futur',
            title="Distribution of Maximum Amount"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_management_page():
    """Display model management page with advanced multi-model support."""
    
    st.header("🔧 Advanced Model Management")
    
    # Get model manager
    model_manager = st.session_state.model_manager
    
    # Tabs for different management functions
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train Models", "📚 Model Library", "📊 Model Comparison", "⚙️ Data Pipeline"])
    
    with tab1:
        st.subheader("Train New Models")
        
        # Model selection for training
        model_options = {
            'linear': '⚡ Linear Regression',
            'neural_network': '🧠 Neural Network',
            'gradient_boost': '🚀 Gradient Boosting'
        }
        
        selected_model = st.selectbox(
            "Choose algorithm to train:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="model_selection"
        )
        
        # Training button
        if st.button("🎯 Train New Model", type="primary", use_container_width=True):
            if st.session_state.dataset is not None:
                train_new_model(selected_model, None)
            else:
                st.error("Dataset not available. Please run the data pipeline first.")
    
    with tab2:
        st.subheader("📚 Saved Models Library")
        
        # List all saved models
        saved_models = model_manager.list_models()
        
        if saved_models:
            # Active model indicator
            active_model = model_manager.get_active_model()
            if active_model:
                active_id = model_manager.active_model_id
                active_info = next((m for m in saved_models if m["model_id"] == active_id), None)
                if active_info:
                    st.success(f"🎯 **Active Model**: {active_info['model_name']} ({active_info['performance_summary']['overall_score']} accuracy)")
            
            st.markdown("---")
            
            # Model cards
            for model in saved_models:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    
                    with col1:
                        is_active = model.get("is_active", False)
                        status_icon = "🎯" if is_active else "📦"
                        st.markdown(f"**{status_icon} {model['model_name']}**")
                        st.caption(f"Type: {model['model_type']} | Created: {model['created_date'][:10]}")
                    
                    with col2:
                        if "performance_summary" in model:
                            perf = model["performance_summary"]
                            st.metric("Checks", perf["checks_accuracy"])
                            st.metric("Amounts", perf["amount_accuracy"])
                    
                    with col3:
                        if "performance_summary" in model:
                            st.metric("Overall", perf["overall_score"])
                        
                        if not is_active:
                            if st.button("🎯 Activate", key=f"activate_{model['model_id']}", use_container_width=True):
                                try:
                                    model_manager.set_active_model(model['model_id'])
                                    st.session_state.prediction_model = model_manager.get_active_model()
                                    st.success(f"✅ Activated: {model['model_name']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to activate model: {e}")
                    
                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{model['model_id']}", use_container_width=True):
                            try:
                                model_manager.delete_model(model['model_id'])
                                if is_active:
                                    st.session_state.prediction_model = None
                                st.success(f"🗑️ Deleted: {model['model_name']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete model: {e}")
                
                st.markdown("---")
        else:
            st.info("📝 No models saved yet. Train your first model in the 'Train Models' tab!")
    
    with tab3:
        st.subheader("📊 Model Performance Comparison")
        
        comparison = model_manager.get_model_comparison()
        
        if comparison["summary"]["total_models"] > 0:
            # Best performers
            st.markdown("### 🏆 Best Performers")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "checks" in comparison["best_performers"]:
                    best = comparison["best_performers"]["checks"]
                    st.metric(
                        "🔢 Best for Checks",
                        best["accuracy"],
                        help=f"Model: {best['model_name']}"
                    )
            
            with col2:
                if "amounts" in comparison["best_performers"]:
                    best = comparison["best_performers"]["amounts"]
                    st.metric(
                        "💰 Best for Amounts",
                        best["accuracy"],
                        help=f"Model: {best['model_name']}"
                    )
            
            with col3:
                if "overall" in comparison["best_performers"]:
                    best = comparison["best_performers"]["overall"]
                    st.metric(
                        "🎯 Best Overall",
                        best["accuracy"],
                        help=f"Model: {best['model_name']}"
                    )
            
            # Performance chart
            if saved_models:
                st.markdown("### 📈 Performance Visualization")
                
                chart_data = []
                for model in saved_models:
                    if "performance_summary" in model:
                        metrics = model["metrics"]
                        chart_data.append({
                            "Model": model["model_name"],
                            "Type": model["model_type"],
                            "Checks Accuracy": metrics.get("nbr_cheques", {}).get("r2", 0) * 100,
                            "Amount Accuracy": metrics.get("montant_max", {}).get("r2", 0) * 100,
                            "Active": "🎯 Active" if model.get("is_active") else "📦 Saved"
                        })
                
                if chart_data:
                    import plotly.express as px
                    import pandas as pd
                    
                    df = pd.DataFrame(chart_data)
                    
                    fig = px.scatter(
                        df,
                        x="Checks Accuracy",
                        y="Amount Accuracy",
                        color="Type",
                        symbol="Active",
                        size=[100] * len(df),
                        hover_data=["Model"],
                        title="Model Performance Comparison",
                        labels={
                            "Checks Accuracy": "Checks Prediction Accuracy (%)",
                            "Amount Accuracy": "Amount Prediction Accuracy (%)"
                        }
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Train some models first to see performance comparisons!")
    
    with tab4:
        st.subheader("⚙️ Data Processing Pipeline")
        
        # Pipeline status
        pipeline_status = check_pipeline_status()
        
        if pipeline_status["completed"]:
            st.success(f"✅ Pipeline completed: {pipeline_status['records']:,} client records processed")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📊 Total Clients", f"{pipeline_status['records']:,}")
                st.metric("🔧 Features", pipeline_status.get('features', 'N/A'))
            
            with col2:
                st.metric("📁 Data Files", f"{pipeline_status.get('files', 'N/A')}")
                st.metric("⏱️ Last Run", pipeline_status.get('last_run', 'N/A'))
        else:
            st.warning("⚠️ Data pipeline not completed")
        
        # Pipeline controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Run Data Pipeline", type="primary", use_container_width=True):
                run_data_pipeline()
        
        with col2:
            if pipeline_status["completed"]:
                if st.button("📊 View Data Statistics", use_container_width=True):
                    show_data_statistics()

def train_new_model(model_type: str, model_name: str = None):
    """Train a new model with the enhanced model manager."""
    model_manager = st.session_state.model_manager
    
    # Show training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Convert dataframe to list of dicts
        status_text.text("📊 Preparing training data...")
        progress_bar.progress(10)
        training_data = st.session_state.dataset.to_dict('records')
        
        # Initialize model with selected type
        status_text.text("🔧 Initializing model...")
        progress_bar.progress(20)
        model = CheckPredictionModel()
        model.set_model_type(model_type)
        
        # Create real-time log container
        log_container = st.empty()
        terminal_logs = []
        
        # Custom stdout capture for real-time updates
        import io
        import contextlib
        import sys
        
        class StreamlitLogger:
            def __init__(self, log_container, terminal_logs, progress_bar, status_text):
                self.log_container = log_container
                self.terminal_logs = terminal_logs
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.original_stdout = sys.stdout
            
            def write(self, text):
                if text.strip() and "[TERMINAL]" in text:
                    self.terminal_logs.append(text.strip())
                    # Update progress based on logs
                    if "TRAINING NUMBER OF CHECKS MODEL" in text:
                        self.progress_bar.progress(30)
                        self.status_text.text("🔵 Training checks prediction model...")
                    elif "TRAINING MAXIMUM AMOUNT MODEL" in text:
                        self.progress_bar.progress(60)
                        self.status_text.text("💰 Training amount prediction model...")
                    elif "RESULTS" in text:
                        self.progress_bar.progress(85)
                        self.status_text.text("📈 Evaluating model performance...")
                    elif "COMPLETED" in text:
                        self.progress_bar.progress(90)
                        self.status_text.text("✅ Training completed!")
                    
                    # Show latest logs
                    recent_logs = self.terminal_logs[-8:]
                    log_text = "\n".join(recent_logs)
                    self.log_container.text_area("🖥️ Training Progress", log_text, height=150)
                
                self.original_stdout.write(text)
            
            def flush(self):
                self.original_stdout.flush()
        
        # Train model with real-time logging
        logger = StreamlitLogger(log_container, terminal_logs, progress_bar, status_text)
        
        model_names = {
            'linear': 'Linear Regression',
            'neural_network': 'Neural Network', 
            'gradient_boost': 'Gradient Boosting'
        }
        
        status_text.text(f"🚀 Training {model_names[model_type]}...")
        with contextlib.redirect_stdout(logger):
            model.fit(training_data)
        
        # Save model with enhanced manager
        status_text.text("💾 Saving model...")
        progress_bar.progress(95)
        
        model_id = model_manager.save_model(model, model_name)
        model_manager.set_active_model(model_id)
        st.session_state.prediction_model = model_manager.get_active_model()
        
        progress_bar.progress(100)
        status_text.text("🎉 Training completed successfully!")
        
        # Success message with model info
        saved_model_info = model_manager.model_registry["models"][model_id]
        st.success(f"✅ Model '{saved_model_info['model_name']}' trained and saved successfully!")
        
        # Show performance metrics
        if hasattr(model, 'metrics') and model.metrics:
            st.markdown("### 📊 Training Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nbr_r2 = model.metrics.get('nbr_cheques', {}).get('r2', 0)
                st.metric(
                    "🔢 Checks Accuracy", 
                    f"{nbr_r2:.1%}",
                    help="How accurately the model predicts number of checks"
                )
            
            with col2:
                amount_r2 = model.metrics.get('montant_max', {}).get('r2', 0)
                st.metric(
                    "💰 Amount Accuracy", 
                    f"{amount_r2:.1%}",
                    help="How accurately the model predicts maximum amounts"
                )
            
            with col3:
                avg_accuracy = (nbr_r2 + amount_r2) / 2
                st.metric(
                    "📈 Overall Score", 
                    f"{avg_accuracy:.1%}",
                    help="Average prediction accuracy across both targets"
                )
        
        # Show training logs
        with st.expander("📋 Complete Training Logs"):
            all_logs = "\n".join(terminal_logs)
            st.text_area("", all_logs, height=200)
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Training failed: {e}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.text(traceback.format_exc())

def check_pipeline_status():
    """Check the status of the data processing pipeline."""
    try:
        dataset_path = Path("data/processed/dataset_final.csv")
        stats_path = Path("data/processed/dataset_statistics.json")
        
        if dataset_path.exists() and stats_path.exists():
            # Load statistics
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            return {
                "completed": True,
                "records": stats.get("dataset_overview", {}).get("total_clients", 0),
                "features": stats.get("dataset_overview", {}).get("total_features", 0),
                "files": len(list(Path("data/processed").glob("*.csv"))) + len(list(Path("data/processed").glob("*.json"))),
                "last_run": dataset_path.stat().st_mtime
            }
        else:
            return {"completed": False}
    except Exception:
        return {"completed": False}

def run_data_pipeline():
    """Run the complete data processing pipeline."""
    with st.spinner("Running complete data processing pipeline..."):
        try:
            builder = DatasetBuilder()
            final_dataset = builder.run_complete_pipeline()
            st.session_state.dataset = pd.DataFrame(final_dataset)
            st.success("✅ Data pipeline completed successfully!")
            st.info(f"📊 Dataset contains {len(final_dataset):,} client records")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")

def show_data_statistics():
    """Show detailed data statistics."""
    try:
        stats_path = Path("data/processed/dataset_statistics.json")
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            st.json(stats)
        else:
            st.warning("Statistics file not found")
    except Exception as e:
        st.error(f"Failed to load statistics: {e}")

if __name__ == "__main__":
    main()