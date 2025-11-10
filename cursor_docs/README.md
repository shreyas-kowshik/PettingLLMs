# PettingLLMs Documentation

This directory contains comprehensive documentation for the entire PettingLLMs codebase.

## 📚 Documentation Files

### 1. [OVERVIEW.md](OVERVIEW.md) - **START HERE!**
Complete system architecture and data flow explanation for all specialization levels (L0/L1/L2/L3).

**Topics Covered**:
- High-level architecture
- End-to-end training step walkthrough
- Detailed comparison of L0, L1, L2, L3 modes
- Configuration inheritance and Hydra usage
- Common issues and solutions

**Read this first** to understand how everything fits together.

---

### 2. [Trainer.md](Trainer.md) - Training System
Deep dive into the `trainer/` module - the core orchestration system.

**Topics Covered**:
- `train.py` - Entry point and Ray initialization
- `MultiAgentsPPOTrainer` - Main training loop
- `MultiAgentsExecutionEngine` - Agent-environment interactions
- `async_generate.py` - LLM API calls and token processing
- PPO update mechanics
- Data flow diagrams
- LoRA differ mode details

**~1,100 lines** of detailed documentation.

---

### 3. [MultiAgentEnv.md](MultiAgentEnv.md) - Environments & Agents
Complete guide to all environment and agent implementations.

**Topics Covered**:
- **Base Classes**: `Env`, `Agent`, `EnvBatch`, `AgentData`
- **Math Environment**: Reasoning + Tool agents, answer verification
- **Code Environment**: Code generation + Unit test agents, execution sandbox
- **Search Environment**: Web search + Reasoning agents, multi-hop QA
- **Stateful Environment**: Planning + Tool agents, stateful APIs
- Ray Docker workers for isolated execution
- Turn order and multi-agent coordination

**~900+ lines** covering all domains.

---

### 4. [Evaluate.md](Evaluate.md) - Evaluation Process
Guide to running inference and evaluation on trained models.

**Topics Covered**:
- `evaluate.py` - Main evaluation logic
- Difference from training mode
- Usage examples for L1, L2, L3
- VLLM server setup
- LoRA adapter loading
- Metrics computation

**Quick reference** for running evaluations.

---

### 5. [Utils.md](Utils.md) - Utility Functions
Documentation of supporting utilities throughout the codebase.

**Topics Covered**:
- `performance.py` - Timing, profiling, colored output
- `logger_config.py` - Multi-logger configuration
- `logging_utils.py` - Structured logging helpers
- `clean_up.py` - Resource cleanup (Ray, temp dirs)

**Essential utilities** used across all modules.

---

## 🗺️ Suggested Reading Order

### For New Users:
1. **OVERVIEW.md** - Understand the big picture
2. **MultiAgentEnv.md** - Learn about specific tasks (Math, Code, etc.)
3. **Trainer.md** - Deep dive into training mechanics
4. **Utils.md** - Reference for common utilities
5. **Evaluate.md** - Run inference on trained models

### For Debugging:
1. **OVERVIEW.md** - Common issues section
2. **Trainer.md** - Training loop details
3. **Utils.md** - Logging and performance monitoring

### For Configuration:
1. **OVERVIEW.md** - L0/L1/L2/L3 comparison and config examples
2. **Trainer.md** - Configuration parameters reference
3. **Evaluate.md** - Evaluation-specific configs

---

## 📊 Quick Reference

### Key Concepts

| Concept | Description | Documentation |
|---------|-------------|---------------|
| **Specialization Levels** | L0/L1/L2/L3 modes for agent differentiation | OVERVIEW.md |
| **PPO Training** | Reinforcement learning algorithm used | Trainer.md |
| **Multi-Agent Interaction** | How agents collaborate | OVERVIEW.md, MultiAgentEnv.md |
| **Environment Domains** | Math, Code, Search, Stateful | MultiAgentEnv.md |
| **LoRA Differ Mode** | Training separate LoRA adapters per agent | Trainer.md, OVERVIEW.md |
| **Ray Workers** | Distributed execution and sandboxing | Trainer.md, MultiAgentEnv.md |
| **Async Rollouts** | Concurrent environment execution | Trainer.md, OVERVIEW.md |

### File Locations

| Component | File Path | Documentation |
|-----------|-----------|---------------|
| **Training Entry** | `pettingllms/trainer/train.py` | Trainer.md |
| **Main Trainer** | `pettingllms/trainer/multi_agents_ppo_trainer.py` | Trainer.md |
| **Execution Engine** | `pettingllms/trainer/multi_agents_execution_engine.py` | Trainer.md |
| **Math Environment** | `pettingllms/multi_agent_env/math/` | MultiAgentEnv.md |
| **Code Environment** | `pettingllms/multi_agent_env/code/` | MultiAgentEnv.md |
| **Evaluation** | `pettingllms/evaluate/evaluate.py` | Evaluate.md |
| **Utilities** | `pettingllms/utils/` | Utils.md |

---

## 🔧 Configuration Examples

### L1: Prompt Mode
```bash
bash scripts/train/math/math_L1_prompt.sh
```
- 1 model shared by all agents
- Agents differentiated by prompts

### L2: LoRA Mode
```bash
bash scripts/train/math/math_L2_lora.sh
```
- 1 base model + separate LoRA adapters per agent
- Best memory efficiency with specialization

### L3: Full Model Mode
```bash
bash scripts/train/math/math_L3_fresh.sh
```
- Separate models for each agent
- Maximum specialization

See **OVERVIEW.md** for detailed comparisons.

---

## 📈 Documentation Statistics

| File | Lines | Focus |
|------|-------|-------|
| **OVERVIEW.md** | ~800 | System architecture, L0/L1/L2/L3 |
| **Trainer.md** | ~1,107 | Training mechanics |
| **MultiAgentEnv.md** | ~900+ | Environments & agents |
| **Evaluate.md** | ~300 | Evaluation |
| **Utils.md** | ~400 | Utilities |
| **Total** | **~3,500+ lines** | Complete codebase coverage |

---

## 💡 Tips

1. **CTRL+F is your friend**: All docs are searchable
2. **Code examples**: Each doc includes practical examples
3. **Cross-references**: Docs reference each other for related topics
4. **Diagrams**: ASCII diagrams for visual understanding
5. **Tables**: Quick reference tables throughout

---

## 🐛 Found an Issue?

If you find any errors or missing information in these docs:
1. Check the actual code implementation
2. Refer to the original research paper
3. Ask in the PettingLLMs community

---

**Last Updated**: November 9, 2025  
**Codebase Version**: PettingLLMs (latest)  
**Documentation Author**: AI Assistant (Claude Sonnet 4.5)

Happy coding! 🎉

