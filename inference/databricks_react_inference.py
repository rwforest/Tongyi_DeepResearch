#!/usr/bin/env python3
"""
Databricks ReAct Inference System
Uses your existing predict() function instead of VLLM servers
Properly migrated version using DatabricksMultiTurnReactAgent and databricks_multi_react
"""

import os
import sys
import json
from typing import List, Dict, Optional, Any, Callable
import argparse
from dotenv import load_dotenv

# Import the properly migrated components
from databricks_react_agent import DatabricksMultiTurnReactAgent
from databricks_multi_react import run_databricks_multi_react


class DatabricksReactInference:
    """
    High-level interface for Databricks ReAct Inference
    Uses the properly migrated DatabricksMultiTurnReactAgent
    """

    def __init__(self, predict_function: Callable[[str], str]):
        """
        Initialize with your predict function

        Args:
            predict_function: Your existing predict() function that takes a string and returns inference
        """
        self.predict_function = predict_function

        # Load environment variables
        load_dotenv()

        print("✅ DatabricksReactInference initialized with predict() function")
        print("   Using properly migrated DatabricksMultiTurnReactAgent and databricks_multi_react")

    def setup_environment(self,
                         dataset: str = "research_dataset",
                         output_path: str = "/tmp/databricks_output",
                         temperature: float = None,
                         presence_penalty: float = None,
                         rollout_count: int = None,
                         max_workers: int = None,
                         # API Keys
                         serper_key: Optional[str] = None,
                         jina_api_keys: Optional[str] = None,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None,
                         summary_model_name: Optional[str] = None,
                         dashscope_api_key: Optional[str] = None,
                         dashscope_api_base: Optional[str] = None,
                         **kwargs) -> Dict[str, str]:
        """
        Set up environment variables for inference
        """

        # Use provided values or defaults from environment
        env_vars = {
            'DATASET': dataset,
            'OUTPUT_PATH': output_path,
            'TEMPERATURE': str(temperature or os.environ.get('TEMPERATURE', 0.85)),
            'PRESENCE_PENALTY': str(presence_penalty or os.environ.get('PRESENCE_PENALTY', 1.1)),
            'ROLLOUT_COUNT': str(rollout_count or os.environ.get('ROLLOUT_COUNT', 3)),
            'MAX_WORKERS': str(max_workers or os.environ.get('MAX_WORKERS', 30))
        }

        # API Keys (only set if provided)
        if serper_key or os.environ.get('SERPER_KEY_ID'):
            env_vars['SERPER_KEY_ID'] = serper_key or os.environ.get('SERPER_KEY_ID')
        if jina_api_keys or os.environ.get('JINA_API_KEYS'):
            env_vars['JINA_API_KEYS'] = jina_api_keys or os.environ.get('JINA_API_KEYS')
        if api_key or os.environ.get('API_KEY'):
            env_vars['API_KEY'] = api_key or os.environ.get('API_KEY')
        if api_base or os.environ.get('API_BASE'):
            env_vars['API_BASE'] = api_base or os.environ.get('API_BASE')
        if summary_model_name or os.environ.get('SUMMARY_MODEL_NAME'):
            env_vars['SUMMARY_MODEL_NAME'] = summary_model_name or os.environ.get('SUMMARY_MODEL_NAME')
        if dashscope_api_key or os.environ.get('DASHSCOPE_API_KEY'):
            env_vars['DASHSCOPE_API_KEY'] = dashscope_api_key or os.environ.get('DASHSCOPE_API_KEY')
        if dashscope_api_base or os.environ.get('DASHSCOPE_API_BASE'):
            env_vars['DASHSCOPE_API_BASE'] = dashscope_api_base or os.environ.get('DASHSCOPE_API_BASE')

        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = str(value)

        print("✅ Environment configured:")
        print(f"   Temperature: {env_vars['TEMPERATURE']}")
        print(f"   Presence Penalty: {env_vars['PRESENCE_PENALTY']}")
        print(f"   Rollout Count: {env_vars['ROLLOUT_COUNT']}")
        print(f"   Max Workers: {env_vars['MAX_WORKERS']}")

        # Print API key status
        api_keys = ['SERPER_KEY_ID', 'JINA_API_KEYS', 'DASHSCOPE_API_KEY']
        for key in api_keys:
            if key in env_vars:
                print(f"   {key}: ✅ Configured")
            else:
                print(f"   {key}: ⚠️ Not configured")

        return env_vars

    def research_question(self,
                         question: str,
                         model_name: str = "tongyi-deep-research") -> Dict[str, Any]:
        """
        Process a single research question using DatabricksMultiTurnReactAgent

        Args:
            question: Research question to process
            model_name: Model identifier

        Returns:
            Result dictionary with prediction and metadata
        """
        print(f"🔬 Processing research question: {question}")

        # Initialize agent
        llm_cfg = {
            'model': model_name,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': float(os.environ.get('TEMPERATURE', 0.85)),
                'top_p': 0.95,
                'presence_penalty': float(os.environ.get('PRESENCE_PENALTY', 1.1))
            },
            'model_type': 'databricks'
        }

        agent = DatabricksMultiTurnReactAgent(
            predict_function=self.predict_function,
            llm=llm_cfg,
            function_list=["search", "visit", "google_scholar", "PythonInterpreter"]
        )

        # Process question
        data = {
            'item': {
                'question': question,
                'answer': ''  # Reference answer (empty for new questions)
            }
        }

        result = agent._run(data, model_name)
        return result

    def batch_research(self,
                      questions: List[str],
                      dataset: str = "research_dataset",
                      output_path: str = "/tmp/databricks_output",
                      model_name: str = "tongyi-deep-research",
                      rollout_count: int = None,
                      max_workers: int = None,
                      **kwargs) -> bool:
        """
        Process multiple research questions using the complete databricks_multi_react system

        Args:
            questions: List of research questions
            dataset: Dataset name
            output_path: Output directory
            model_name: Model identifier
            rollout_count: Number of rollouts per question
            max_workers: Maximum worker threads
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        print(f"📚 Processing {len(questions)} research questions...")

        # Create temporary dataset file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for question in questions:
                item = {
                    'question': question,
                    'answer': ''  # Empty reference answer
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_data_file = f.name

        try:
            # Run complete system
            success = run_databricks_multi_react(
                predict_function=self.predict_function,
                dataset=dataset,
                output_path=output_path,
                model_name=model_name,
                temperature=float(os.environ.get('TEMPERATURE', 0.85)),
                presence_penalty=float(os.environ.get('PRESENCE_PENALTY', 1.1)),
                rollout_count=rollout_count or int(os.environ.get('ROLLOUT_COUNT', 3)),
                max_workers=max_workers or int(os.environ.get('MAX_WORKERS', 30)),
                data_file=temp_data_file,
                **kwargs
            )

            return success

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_data_file)
            except OSError:
                pass

    def run_complete_inference(self,
                              questions: Optional[List[str]] = None,
                              dataset: str = "research_dataset",
                              output_path: str = "/tmp/databricks_output",
                              model_name: str = "tongyi-deep-research",
                              data_file: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Run complete ReAct inference pipeline - equivalent to run_react_infer.sh

        Args:
            questions: List of questions to process (optional if data_file provided)
            dataset: Dataset name
            output_path: Output directory
            model_name: Model identifier
            data_file: Path to existing data file
            **kwargs: Additional parameters

        Returns:
            Dictionary with results and status information
        """
        print("🚀 Starting Complete Databricks ReAct Inference Pipeline")
        print("=" * 60)

        result = {
            "status": "started",
            "dataset": dataset,
            "output_path": output_path,
            "model_name": model_name,
            "questions_processed": 0,
            "success": False,
            "error": None
        }

        try:
            # Setup environment
            self.setup_environment(
                dataset=dataset,
                output_path=output_path,
                **kwargs
            )

            # Use provided questions or data file
            if questions and not data_file:
                success = self.batch_research(
                    questions=questions,
                    dataset=dataset,
                    output_path=output_path,
                    model_name=model_name,
                    **kwargs
                )
                result["questions_processed"] = len(questions)
            else:
                # Use run_databricks_multi_react directly
                success = run_databricks_multi_react(
                    predict_function=self.predict_function,
                    dataset=dataset,
                    output_path=output_path,
                    model_name=model_name,
                    data_file=data_file,
                    **kwargs
                )
                result["questions_processed"] = "See output files"

            if success:
                result["status"] = "completed"
                result["success"] = True
                print("✅ Complete ReAct inference pipeline completed successfully!")
            else:
                result["status"] = "failed"
                result["error"] = "Pipeline execution failed"
                print("❌ Pipeline execution failed")

        except Exception as e:
            print(f"❌ Error during inference: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result


def create_predict_wrapper(logger, context=None):
    """
    Create a predict function wrapper for your MLflow model

    Args:
        logger: Your HuggingFaceMLflowLogger instance
        context: MLflow context (optional)

    Returns:
        Function that takes a string and returns prediction
    """
    def predict_function(prompt: str) -> str:
        model_input = [{
            "prompt": prompt,
            "max_length": 2048,
            "temperature": float(os.environ.get('TEMPERATURE', 0.85)),
            "presence_penalty": float(os.environ.get('PRESENCE_PENALTY', 1.1))
        }]

        try:
            results = logger.predict(context, model_input)
            if isinstance(results, list) and len(results) > 0:
                return str(results[0])
            else:
                return str(results)
        except Exception as e:
            return f"Error in prediction: {e}"

    return predict_function


def main():
    """Command line interface for Databricks ReAct inference"""
    parser = argparse.ArgumentParser(
        description="Databricks ReAct Inference using your predict() function"
    )

    parser.add_argument("--questions", nargs='+', help="Research questions to answer")
    parser.add_argument("--dataset", default="research_dataset", help="Dataset name")
    parser.add_argument("--output_path", default="/tmp/databricks_output", help="Output directory")
    parser.add_argument("--model_name", default="tongyi-deep-research", help="Model name")
    parser.add_argument("--data_file", help="Path to data file")
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
    parser.add_argument("--presence_penalty", type=float, default=1.1, help="Presence penalty")
    parser.add_argument("--rollout_count", type=int, default=3, help="Rollout count")
    parser.add_argument("--max_workers", type=int, default=30, help="Max workers")

    # API Keys
    parser.add_argument("--serper_key", help="Serper API key")
    parser.add_argument("--jina_api_keys", help="Jina API keys")
    parser.add_argument("--dashscope_api_key", help="Dashscope API key")

    args = parser.parse_args()

    print("🚀 Databricks ReAct Inference")
    print("=" * 50)
    print("⚠️ Note: This requires integration with your predict() function")
    print()

    if args.questions:
        print(f"Questions to process: {args.questions}")
        print()
        print("To run this, you need to:")
        print("1. Import your HuggingFaceMLflowLogger")
        print("2. Create predict_function = create_predict_wrapper(logger)")
        print("3. Initialize inference = DatabricksReactInference(predict_function)")
        print("4. Call inference.run_complete_inference(questions=args.questions)")
    else:
        print("No questions provided. Use --questions or --data_file")


if __name__ == "__main__":
    main()