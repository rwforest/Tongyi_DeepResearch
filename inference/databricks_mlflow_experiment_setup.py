#!/usr/bin/env python3
"""
Databricks MLflow Experiment Setup for Tongyi DeepResearch
Shows how to properly configure and pass MLflow experiment settings to all modules
"""

import os
import sys
import mlflow

def setup_databricks_mlflow_experiment():
    """
    Complete example of setting up MLflow experiment for Databricks Unity Catalog
    This configuration will be passed to all modules automatically
    """

    print("🏗️ DATABRICKS MLFLOW EXPERIMENT SETUP")
    print("=" * 60)

    # =========================================================================
    # 1. DATABRICKS UNITY CATALOG CONFIGURATION
    # =========================================================================

    # Configure your Databricks Unity Catalog settings
    EXP_NAME = "/Users/first.last@databricks.com/tongyi_deepresearch_experiment"
    CATALOG = "my_catalog"
    SCHEMA = "my_schema"
    VOLUME = "my_volume"
    ARTIFACT_PATH = f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

    print(f"📊 Experiment Configuration:")
    print(f"   Name: {EXP_NAME}")
    print(f"   Catalog: {CATALOG}")
    print(f"   Schema: {SCHEMA}")
    print(f"   Volume: {VOLUME}")
    print(f"   Artifact Path: {ARTIFACT_PATH}")

    # =========================================================================
    # 2. MLFLOW CONFIGURATION SETUP
    # =========================================================================

    try:
        # Import the global configuration manager
        from mlflow_config import set_global_mlflow_config, ensure_mlflow_initialized

        # Set MLflow URIs for Databricks
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        print("✅ MLflow URIs configured for Databricks")

        # Create experiment if it doesn't exist
        if mlflow.get_experiment_by_name(EXP_NAME) is None:
            experiment_id = mlflow.create_experiment(
                name=EXP_NAME,
                artifact_location=ARTIFACT_PATH
            )
            print(f"✅ Created experiment: {EXP_NAME} (ID: {experiment_id})")
        else:
            print(f"✅ Using existing experiment: {EXP_NAME}")

        # Set the active experiment
        mlflow.set_experiment(EXP_NAME)
        print(f"✅ Active experiment set to: {EXP_NAME}")

        # =========================================================================
        # 3. CONFIGURE GLOBAL MLFLOW SETTINGS (KEY PART!)
        # =========================================================================

        # This sets the global configuration that ALL modules will use
        set_global_mlflow_config(
            experiment_name=EXP_NAME,
            catalog=CATALOG,
            schema=SCHEMA,
            volume=VOLUME,
            artifact_location=ARTIFACT_PATH,
            tracking_uri="databricks",
            registry_uri="databricks-uc"
        )
        print("✅ Global MLflow configuration set - all modules will use this!")

        # Ensure initialization
        ensure_mlflow_initialized()

        return True

    except Exception as e:
        print(f"❌ MLflow setup failed: {e}")
        return False

def demonstrate_experiment_propagation():
    """
    Demonstrate how the experiment configuration propagates to all modules
    """

    print(f"\n🔄 EXPERIMENT PROPAGATION DEMONSTRATION")
    print("=" * 60)

    try:
        # Import modules that use MLflow - they will automatically use global config
        from databricks_react_agent import DatabricksMultiTurnReactAgent
        from traced_predict_wrapper import create_traced_predict_function

        print("✅ Imported modules - they automatically use global MLflow config")

        # Create a demo predict function
        def demo_predict(model_input):
            """Demo predict function for testing"""
            import time
            time.sleep(0.1)  # Simulate processing
            if isinstance(model_input, list) and len(model_input) > 0:
                prompt = model_input[0].get('prompt', str(model_input))
            else:
                prompt = str(model_input)
            return f"Demo response to: {prompt[:50]}..."

        # Create traced predict function - uses global MLflow config
        traced_predict = create_traced_predict_function(
            demo_predict,
            model_name="tongyi-deepresearch-demo"
        )
        print("✅ Created traced predict function - uses global experiment")

        # Create ReAct agent - uses global MLflow config
        agent = DatabricksMultiTurnReactAgent(
            predict_function=traced_predict,
            llm={'model': 'demo-model'}
        )
        print("✅ Created ReAct agent - uses global experiment")

        # Test with a research session - everything traces to your experiment
        test_data = {
            'item': {
                'question': 'What is the global MLflow configuration?',
                'answer': 'Demo answer'
            }
        }

        print(f"\n🧪 Testing complete workflow with global experiment...")

        # This creates a run in YOUR configured experiment
        with mlflow.start_run(run_name="demo_global_config_test") as run:
            print(f"📊 Started run in experiment: {mlflow.get_experiment(run.info.experiment_id).name}")

            # All nested spans will be in the same experiment
            result = agent._run(test_data)

            print(f"✅ Research completed in your Databricks experiment!")
            print(f"📊 Run ID: {run.info.run_id}")
            print(f"📁 Artifacts stored in: {ARTIFACT_PATH}")

    except Exception as e:
        print(f"⚠️ Demo warning: {e}")
        print("   This is normal if dependencies aren't fully configured")

def alternative_configuration_methods():
    """
    Show alternative ways to configure MLflow experiment
    """

    print(f"\n🛠️ ALTERNATIVE CONFIGURATION METHODS")
    print("=" * 60)

    print("🔧 Method 1: Environment Variables")
    print("   Set these in your notebook or environment:")
    print("   ```")
    print("   os.environ['MLFLOW_EXPERIMENT_NAME'] = '/Users/you@company.com/your_experiment'")
    print("   os.environ['MLFLOW_TRACKING_URI'] = 'databricks'")
    print("   os.environ['MLFLOW_REGISTRY_URI'] = 'databricks-uc'")
    print("   os.environ['DATABRICKS_CATALOG'] = 'your_catalog'")
    print("   os.environ['DATABRICKS_SCHEMA'] = 'your_schema'")
    print("   os.environ['DATABRICKS_VOLUME'] = 'your_volume'")
    print("   ```")

    print("\n🔧 Method 2: Direct Function Calls")
    print("   Call this in your notebook before using any modules:")
    print("   ```python")
    print("   from mlflow_config import set_global_mlflow_config")
    print("   set_global_mlflow_config(")
    print("       experiment_name='/Users/you@company.com/your_experiment',")
    print("       catalog='your_catalog',")
    print("       schema='your_schema',")
    print("       volume='your_volume'")
    print("   )")
    print("   ```")

    print("\n🔧 Method 3: Configuration File")
    print("   Add to your .env file:")
    print("   ```")
    print("   MLFLOW_EXPERIMENT_NAME=/Users/you@company.com/your_experiment")
    print("   MLFLOW_TRACKING_URI=databricks")
    print("   MLFLOW_REGISTRY_URI=databricks-uc")
    print("   DATABRICKS_CATALOG=your_catalog")
    print("   DATABRICKS_SCHEMA=your_schema")
    print("   DATABRICKS_VOLUME=your_volume")
    print("   ```")

def show_experiment_verification():
    """
    Show how to verify the experiment configuration is working
    """

    print(f"\n✅ EXPERIMENT CONFIGURATION VERIFICATION")
    print("=" * 60)

    try:
        from mlflow_config import get_global_mlflow_config

        # Get current configuration
        config = get_global_mlflow_config()

        print("📊 Current Global MLflow Configuration:")
        for key, value in config.items():
            if value is not None:
                print(f"   {key}: {value}")

        # Verify active experiment
        try:
            active_experiment = mlflow.get_experiment_by_name(config['experiment_name'])
            if active_experiment:
                print(f"\n✅ Active Experiment Verified:")
                print(f"   Name: {active_experiment.name}")
                print(f"   ID: {active_experiment.experiment_id}")
                print(f"   Artifact Location: {active_experiment.artifact_location}")
            else:
                print(f"\n⚠️ Experiment not found: {config['experiment_name']}")
        except Exception as e:
            print(f"\n⚠️ Could not verify experiment: {e}")

    except Exception as e:
        print(f"❌ Verification failed: {e}")

def main():
    """
    Complete example of Databricks MLflow experiment setup
    """

    print("🚀 DATABRICKS MLFLOW EXPERIMENT SETUP GUIDE")
    print("=" * 80)

    # Step 1: Setup experiment
    success = setup_databricks_mlflow_experiment()

    if success:
        # Step 2: Demonstrate propagation
        demonstrate_experiment_propagation()

        # Step 3: Show alternatives
        alternative_configuration_methods()

        # Step 4: Verification
        show_experiment_verification()

        print(f"\n" + "=" * 80)
        print("🎯 SUMMARY")
        print("✅ Databricks MLflow experiment configured successfully!")
        print("✅ Global configuration set - all modules will use your experiment")
        print("✅ Unity Catalog integration enabled")
        print("✅ Artifacts will be stored in your specified volume")

        print(f"\n💡 NEXT STEPS:")
        print("1. Use this setup code in your Databricks notebook")
        print("2. Modify the experiment name, catalog, schema, and volume")
        print("3. Run your ReAct research sessions")
        print("4. View traces in your experiment: Experiments → Your Experiment")

    else:
        print(f"\n❌ Setup failed - check your Databricks configuration")

if __name__ == "__main__":
    main()