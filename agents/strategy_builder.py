import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class StrategyBuilder:
    def __init__(self):
        # We allow a slightly higher temperature (0.2) here because we want 
        # strategic brainstorming, but still grounded in reality.
        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_strategy(self, sector: str, turnover: float, primary_activity: str) -> dict:
        """
        Generates a corporate tax restructuring strategy and calculated mock projections.
        """
        prompt_template = """
        You are a highly sought-after Indian Corporate Tax Strategist and Chartered Accountant operating in 2026.
        Your client is an enterprise in the {sector} sector, primarily engaged in {primary_activity}, with an annual turnover of ₹{turnover} Cr.
        
        Provide 2 highly specific, actionable tax restructuring strategies they can use to legally reduce their corporate tax burden or optimize their GST/ITC structure under current Indian tax laws.
        
        Format the output EXACTLY as a valid JSON object with the following keys. Do not output anything else.
        {{
            "strategy_markdown": "A formatted markdown string explaining the 2 strategies with headers and bullet points.",
            "estimated_savings_percentage": 15
        }}
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        try:
            # Execute the prompt
            response = chain.invoke({
                "sector": sector,
                "turnover": turnover,
                "primary_activity": primary_activity
            })
            
            raw_output = response.content.strip()
            
            # --- NEW ROBUST JSON EXTRACTION ---
            # Find the first '{' and the last '}' to ignore any conversational text
            start_idx = raw_output.find('{')
            end_idx = raw_output.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = raw_output[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON object found in the LLM response.")
            
        except Exception as e:
            # Fallback in case of severe API or parsing failure
            print(f"Strategy Error: {str(e)}") # This will print to your VS Code terminal so you can see what went wrong
            return {
                "strategy_markdown": f"### ⚠️ Processing Error\nThe AI encountered an issue structuring the strategy. \n\n*Technical Details for Debugging: {str(e)}*",
                "estimated_savings_percentage": 0
            }