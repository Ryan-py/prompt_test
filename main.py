import gradio as gr
import time
import requests
import json
import random

# --- LLM API Configuration ---

apiKey = "" 

GEMINI_MODEL_MAP = {
    "Gemini 2.5 Flash": "gemini-2.5-flash-preview-09-2025",
    "Gemini 2.5 Pro": "gemini-2.5-pro-preview-09-2025",
}

# --- LLM API Call and Performance Calculation ---

def run_test(
    system_prompt: str,
    user_prompt: str,
    model_name_ui: str,
    temperature: float,
    top_p: float,
    max_tokens: int
):
    """
    Calls the Gemini API, extracts the response, and calculates performance metrics.
    
    Returns:
        tuple: (response_text, metrics_markdown)
    """
    
    model_name_api = GEMINI_MODEL_MAP.get(model_name_ui, "gemini-2.5-flash-preview-09-2025")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name_api}:generateContent?key={apiKey}"

    # Construct the request payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens
        }
    }
    
    max_retries = 3
    
    # 1. Start timer
    start_time = time.time()

    for attempt in range(max_retries):
        try:
            # Make the API request
            response = requests.post(
                api_url, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=45 # Increased timeout for potential long generations
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract content and metrics
            # Check for text in the first candidate part
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 
                "Model returned no text, check the console for safety filters or other issues.")
            
            # Token usage from usageMetadata
            usage_metadata = result.get('usageMetadata', {})
            input_tokens = usage_metadata.get('promptTokenCount', 0)
            output_tokens = usage_metadata.get('candidatesTokenCount', 0)
            total_tokens = usage_metadata.get('totalTokenCount', 0)

            # 2. End timer and calculate latency
            latency = round(time.time() - start_time, 2)
            
            # --- Cost Calculation (Illustrative Rates) ---
            # NOTE: These rates are illustrative and based on a typical tiered pricing structure.
            if "flash" in model_name_api:
                # Example Flash rates: Input $0.00035 / 1K tokens, Output $0.0007 / 1K tokens
                input_cost = (input_tokens / 1000) * 0.00035
                output_cost = (output_tokens / 1000) * 0.0007
            elif "pro" in model_name_api:
                # Example Pro rates: Input $0.0035 / 1K tokens, Output $0.007 / 1K tokens
                input_cost = (input_tokens / 1000) * 0.0035
                output_cost = (output_tokens / 1000) * 0.007
            else:
                input_cost = 0.0
                output_cost = 0.0

            total_cost = round(input_cost + output_cost, 6)

            # Build the Markdown string for performance metrics
            metrics_markdown = f"""
            ### 📊 Performance Summary
            | Metric | Value | Details |
            | :--- | :--- | :--- |
            | **Model Used** | `{model_name_api}` | The specific API model called. |
            | **Latency** | `{latency:.2f} s` | Total time for API response. |
            | **Input Tokens** | `{input_tokens:,}` | Tokens in System + User Prompts. |
            | **Output Tokens** | `{output_tokens:,}` | Tokens in Model Response. |
            | **Total Tokens** | `{total_tokens:,}` | Sum of Input and Output. |
            | **Simulated Cost** | `${total_cost:.6f}` | Based on illustrative per-token rates. |
            """
            
            return text, metrics_markdown

        except requests.exceptions.RequestException as e:
            # Handle HTTP errors, timeouts, and connection issues with exponential backoff
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"API call failed: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time + random.uniform(0, 0.5)) # Add jitter
            else:
                error_message = f"API Error after {max_retries} attempts: {e}. Check API key and network."
                metrics_markdown = f"### ❌ Error\nCould not connect to or receive a valid response from the Gemini API. Details: `{error_message}`"
                return f"Error: {error_message}", metrics_markdown
        except Exception as e:
             # Handle JSON parsing or other general errors
            error_message = f"Unexpected Error: {e}. Check console for details."
            metrics_markdown = f"### ❌ Error\nUnexpected Error during processing: `{error_message}`"
            return f"Error: {error_message}", metrics_markdown

    # Fallback return in case loop completes unexpectedly
    return "Unknown Error", "### ❌ Error\nAn unhandled error occurred."


# --- Gradio Interface Layout (using Blocks for custom design) ---

with gr.Blocks(title="Team Prompt Testing Platform", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🚀 LLM Prompt Engineering Sandbox
        A platform for your team to test different prompt techniques and track performance metrics.
        
        **Instructions:** Fill out the prompts, adjust the parameters, and click **Run Test** to see the response and performance data.
        """
    )

    with gr.Row(variant="panel"):
        
        # --- LEFT COLUMN: Inputs and Parameters ---
        with gr.Column(scale=1):
            
            gr.Markdown("## 📝 Prompts & Model Selection")

            model_dropdown = gr.Dropdown(
                choices=list(GEMINI_MODEL_MAP.keys()),
                label="Model Target",
                value="Gemini 2.5 Flash",
                allow_custom_value=False
            )

            system_prompt = gr.Textbox(
                label="System Prompt / Persona",
                placeholder="E.g., Act as a senior software architect specializing in cloud infrastructure.",
                lines=4
            )

            user_prompt = gr.Textbox(
                label="User Query / Task",
                placeholder="Paste your specific task, few-shot examples, or chain-of-thought instructions here...",
                lines=8
            )

            gr.Markdown("## ⚙️ Generation Parameters")
            
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.7,
                    label="Temperature (Randomness)"
                )
                
                top_p_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.9,
                    label="Top P (Diversity Sampling)"
                )

            max_tokens_number = gr.Number(
                label="Max Output Tokens",
                value=2048,
                minimum=128,
                maximum=8192,
                step=64
            )

            # --- RUN BUTTON ---
            run_button = gr.Button("▶️ Run Test", variant="primary", scale=0)
            
        # --- RIGHT COLUMN: Output and Performance ---
        with gr.Column(scale=2):
            
            gr.Markdown("## 🤖 Model Response")

            response_output = gr.Textbox(
                label="Generated Output",
                lines=20,
                interactive=False,
                autoscroll=True
            )
            
            performance_output = gr.Markdown("---")


    # --- Define Interactions ---
    run_button.click(
        fn=run_test,
        inputs=[
            system_prompt, 
            user_prompt, 
            model_dropdown, 
            temperature_slider, 
            top_p_slider, 
            max_tokens_number
        ],
        outputs=[response_output, performance_output]
    )

# Launch the Gradio App
demo.launch()
# NOTE: The launch command is commented out for compatibility with the execution environment.
# To run this locally, uncomment the last line and execute the Python file.
