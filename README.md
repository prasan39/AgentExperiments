# Agent Experiments 🤖

**Code repository for Tech-tonic Chronicles' agentic AI explorations**

---

## **About Tech-tonic Chronicles**

Welcome aboard the journey of Prasanna's Tech-tonic Chronicles! We are living through an exciting era where technology is rapidly evolving, enabling new ways to tackle challenges and create value. I am starting this newsletter as my space to share my ongoing journey of exploring these advancements, focusing on the practical insights I gain from **learning and building with them**.

Drawing from years of experience advising CXOs and working with various teams across different sectors, you can expect practical insights, experiments, and real-world examples, not just theory. Initial focus of the chronicles will be on the transformative landscape of AI.

My goal is to help you understand and leverage these 'tech-tonic' shifts through practical application. I would love to hear your thoughts and feedback - let us learn, build, and share insights together.

---

## **This Repository**

This repository contains the hands-on code experiments from my Tech-tonic Chronicles newsletter, focusing on **agentic AI systems**. Each experiment demonstrates practical implementations of AI agent frameworks, multi-agent orchestration, and production-ready patterns.

The code here bridges the gap between theoretical concepts and real-world applications, showing how to build from simple prototypes to scalable AI systems.

---

## **Current Experiments**

### **Semantic Kernel Demos** 
*From Tech-tonic Chronicles #13: "From Solo Agents to AI Agent Teams"*

#### **1. semantic_kernel_agent_demo.py**
A foundational single-agent implementation that demonstrates:
- **Azure OpenAI Integration**: Seamless connection to Azure OpenAI services
- **Agent Wrapper Pattern**: How Semantic Kernel wraps AI models as structured agents
- **Basic Orchestration**: Simple request-response flow with built-in conversation management
- **Production Foundation**: Clean, reusable code structure for enterprise applications

**Key Features:**
- Environment-based configuration
- Error handling and cleanup
- Conversation state management through AgentGroupChat
- One-turn and multi-turn conversation support

#### **2. semantic_kernel_groupchat_demo.py**
A multi-agent orchestration system showcasing:
- **Specialized Agent Roles**: Planner, Researcher, and Editor agents with distinct personas
- **Group Chat Orchestration**: Agents collaborate through structured conversation patterns
- **Dynamic Turn Management**: SK's AgentGroupChat automatically manages who speaks next
- **Termination Strategy**: Intelligent conversation completion based on task fulfillment
- **Real-time Streaming**: Live updates as agents contribute to the workflow

**Key Features:**
- Three distinct agent personas with specialized instructions
- Configurable maximum conversation rounds
- Streaming generator for real-time updates
- Role-based color coding for output visualization
- Enterprise-ready error handling

**Agent Personas:**
- **Planner**: Strategic thinking and project coordination
- **Researcher**: Information gathering and analysis  
- **Editor**: Content refinement and quality assurance

---

## **Getting Started**

### **Prerequisites**
```bash
pip install semantic-kernel python-dotenv openai
```

### **Environment Setup**
Create a `.env` file with:
```
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### **Running the Demos**
```bash
# Single agent demo
python semantic_kernel_agent_demo.py

# Multi-agent group chat demo  
python semantic_kernel_groupchat_demo.py
```

---

## **Learning Path**

These experiments follow a progressive learning approach:

1. **Foundation**: Start with single-agent patterns (`semantic_kernel_agent_demo.py`)
2. **Orchestration**: Progress to multi-agent collaboration (`semantic_kernel_groupchat_demo.py`)
3. **Scaling**: Build toward production-ready systems with observability and error handling

Each demo includes detailed comments and follows enterprise development patterns, making them suitable for both learning and production adaptation.

---

## **Connect & Contribute**

**Newsletter**: https://lnkd.in/p/gFDFZNza

Found these experiments helpful? Have ideas for new agentic patterns to explore? Let's learn and build together!

---

*"From prototype to production - exploring the practical side of agentic AI, one experiment at a time."*
