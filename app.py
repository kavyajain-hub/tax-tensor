import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
from agents.decoder_agent import DecoderAgent
from core.vectorstore import VectorStoreManager
from agents.itc_auditor import ITCAuditor
from agents.strategy_builder import StrategyBuilder
from dotenv import load_dotenv
load_dotenv()
# Page Config
st.set_page_config(page_title="TAX-TENSOR | AI Tax & GST Auditor", layout="wide", page_icon="⚖️")

# Initialize Session State
if "indexed_circulars" not in st.session_state:
    st.session_state.indexed_circulars = []
if "tax_liability" not in st.session_state:
    st.session_state.tax_liability = 1450000 # Default starting value
if "itc_risk" not in st.session_state:
    st.session_state.itc_risk = 0 # Starts at 0 until gaps are detected
if "itc_delta" not in st.session_state:
    st.session_state.itc_delta = 0


# Title & Sidebar
st.title("⚖️ TAX-TENSOR")
st.caption("AI Corporate Tax Restructuring & GST Auditor")

with st.sidebar:
    st.header("📊 Compliance Overview")
    
    # Dynamic Metrics
    st.metric(
        label="Estimated Tax Liability", 
        value=f"₹ {st.session_state.tax_liability:,.0f}"
    )
    
    # Only show delta if there is a change
    delta_str = f"-₹ {st.session_state.itc_delta:,.0f} Action Required" if st.session_state.itc_delta > 0 else None
    
    st.metric(
        label="ITC at Risk", 
        value=f"₹ {st.session_state.itc_risk:,.0f}", 
        delta=delta_str, 
        delta_color="inverse"
    )

# Main Workspace Tabs
tab1, tab2, tab3 = st.tabs(["📜 Notification Decoder", "🔍 ITC Gap Detector", "🏗️ Tax Strategy Builder"])

# --- TAB 1: NOTIFICATION DECODER ---
with tab1:
    st.header("Translate GST Circulars to Business Impact")
    uploaded_circular = st.file_uploader("Upload CBIC/GST Notification (PDF)", type=["pdf"])
    
    # 1. Indexing the PDF
    if uploaded_circular:
        if st.button("Index Notification"):
            with st.spinner("Parsing legal clauses and embedding data locally..."):
                try:
                    # Save file temporarily for the vectorstore to process
                    save_path = os.path.join("data", "circulars", uploaded_circular.name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_circular.getbuffer())
                    
                    # Execute indexing
                    vsm = VectorStoreManager()
                    vsm.index_document(save_path, collection_name="circulars")
                    
                    # Check session state for deduplication UI feedback
                    if uploaded_circular.name not in st.session_state.indexed_circulars:
                        st.session_state.indexed_circulars.append(uploaded_circular.name)
                        st.success(f"Successfully vectorized and indexed {uploaded_circular.name}!")
                    else:
                        # NEW: Graceful feedback if they upload the same file twice
                        st.info(f"'{uploaded_circular.name}' is already indexed in the database. No duplicate embeddings created.")
                        
                except Exception as e:
                    # NEW: Catch the SQLite/PDF errors we explicitly raised in vectorstore.py
                    st.error(f"System Error during indexing: {str(e)}")
            
    st.markdown("---")
    st.markdown("### Ask the AI Tax Auditor")
    
    # We use a form here. Forms force Streamlit to wait until the user explicitly 
    # clicks submit before re-running the app. This solves 99% of input bugs.
    is_indexed = len(st.session_state.indexed_circulars) > 0
    
    with st.form(key="rag_query_form"):
        query = st.text_input(
            label="Type your question below:",
            placeholder="Upload and index a notification above to start asking questions..." if not is_indexed else "e.g., How does this circular impact ITC claims?",
            disabled=not is_indexed
        )
        
        # The submit button acts as our strict trigger
        submit_button = st.form_submit_button(label="Analyze Legal Impact", disabled=not is_indexed)
    
    # 2. Querying the LLM (Only runs when the button is clicked)
    if submit_button and query:
        if not st.session_state.indexed_circulars:
            st.error("Please upload and index a GST Notification PDF first.")
        else:
            with st.spinner("Cross-referencing legal framework via Groq (Llama 3.3)..."):
                try:
                    # Force load env variables inside the execution block just to be safe
                    from dotenv import load_dotenv
                    load_dotenv()
                    
                    agent = DecoderAgent()
                    answer = agent.analyze_impact(query)
                    
                    # Display the structured markdown returned by the agent
                    st.success("Analysis Complete!")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error connecting to LLM: {str(e)}")
                    
    elif submit_button and not query:
        st.warning("Please type a question before submitting.")
# --- TAB 2: ITC GAP DETECTOR ---
with tab2:
    st.header("Automated Input Tax Credit (ITC) Reconciliation")
    st.write("Upload your internal purchase register and the government auto-drafted GSTR-2B to detect unmatched invoices and prevent ITC denial.")
    
    col1, col2 = st.columns(2)
    with col1:
        internal_register = st.file_uploader("Upload Internal Purchase Register (CSV/Excel)", type=["csv", "xlsx"], key="internal")
    with col2:
        gstr_2b = st.file_uploader("Upload GSTR-2B (CSV/Excel)", type=["csv", "xlsx"], key="gstr")
        
    if internal_register and gstr_2b and st.button("Run Reconciliation Engine"):
        with st.spinner("Executing fuzzy matching on Invoice Numbers and GSTINs..."):
            
            # Helper to read CSV or Excel natively
            def load_data(file):
                if file.name.endswith('.csv'):
                    return pd.read_csv(file)
                else:
                    return pd.read_excel(file)

            try:
                # 1. Load Data
                int_df = load_data(internal_register)
                g2b_df = load_data(gstr_2b)

                # 2. Run Engine
                auditor = ITCAuditor()
                anomalies = auditor.reconcile(int_df, g2b_df)

                # 3. Display Results
                st.subheader("⚠️ High-Risk Anomalies Detected")
                if anomalies.empty:
                    st.success("No discrepancies found! All invoices and values match perfectly.")
                    st.session_state.itc_risk = 0 # Reset risk
                else:
                    # Update the sidebar state dynamically!
                    total_risk = anomalies['ITC at Risk'].sum()
                    st.session_state.itc_delta = total_risk - st.session_state.itc_risk # Calculate change
                    st.session_state.itc_risk = total_risk
                    st.metric("Total ITC Value at Risk", f"₹ {total_risk:,.2f}")
                    
                    # Display the final table
                    st.dataframe(anomalies, use_container_width=True, hide_index=True)
                    
                    st.markdown("### 🤖 AI Recommended Action")
                    st.write("Review the **🔴 HIGH** severity items immediately. These are invoices you have recorded and paid, but the vendor has failed to upload them to the GST portal. You cannot claim this ITC until they comply.")

            except KeyError as e:
                st.error(f"Column Mapping Error: The system could not find a required column. Please ensure both files contain: 'Gstin', 'Invoice Number', and 'Tax Amount'.")
            except Exception as e:
                st.error(f"An error occurred during reconciliation: {str(e)}")

# --- TAB 3: TAX STRATEGY BUILDER ---
with tab3:
    st.header("Corporate Tax Restructuring & Optimization")
    st.write("Input your business parameters to generate legally compliant tax-saving strategies and corporate structuring advice.")
    
    # User Input Form
    with st.form("strategy_form"):
        col1, col2 = st.columns(2)
        with col1:
            biz_sector = st.selectbox("Industry Sector", [
                "Logistics & Warehousing", 
                "IT / ITES Data Centers", 
                "Manufacturing", 
                "Healthcare & Pharma",
                "E-Commerce"
            ])
            turnover = st.number_input("Annual Turnover (₹ Crores)", min_value=1.0, value=15.0, step=1.0)
        with col2:
            primary_activity = st.text_area("Primary Business Activity", value="Providing inter-state freight transport and managing cold-storage warehouses.", height=110)
            
        submitted = st.form_submit_button("Generate Optimization Strategy")
        
    if submitted:
        with st.spinner("Analyzing corporate structures and projecting tax savings..."):
            time.sleep(1) # Simulating complex processing
            
            # Run the Strategy Agent
            builder = StrategyBuilder()
            result = builder.generate_strategy(biz_sector, turnover, primary_activity)
            
            # Calculate mock financial projections based on LLM output
            savings_pct = result.get("estimated_savings_percentage", 15)
            
            # Simple assumption: Tax liability is roughly 25% of 15% net margin of turnover
            current_liability = (turnover * 10000000) * 0.15 * 0.25 
            optimized_liability = current_liability * (1 - (savings_pct / 100))
            
            st.markdown("---")
            st.subheader("📊 Projected Tax Liability Impact")
            
            # Visualization using Plotly
            chart_data = pd.DataFrame({
                "Scenario": ["Current Structure", "Optimized Structure"],
                "Tax Liability (₹)": [current_liability, optimized_liability]
            })
            
            fig = px.bar(
                chart_data, 
                x="Scenario", 
                y="Tax Liability (₹)", 
                color="Scenario",
                text_auto='.2s',
                color_discrete_sequence=["#EF553B", "#00CC96"],
                title=f"Potential Liability Reduction: {savings_pct}%"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the AI-generated strategy
            st.markdown("### 🏗️ AI Strategic Recommendations")
            st.markdown(result.get("strategy_markdown", ""))
            
            # Download Button for the Report
            st.download_button(
                label="📥 Download Strategy Report (PDF)",
                data="Mock PDF Payload",
                file_name=f"Tax_Strategy_{biz_sector.replace(' ', '')}.pdf",
                mime="application/pdf"
            )