"""
Day 99: Portfolio Planning - Interactive Portfolio Planner
"""
import json
from typing import Dict, List


class PortfolioPlanner:
    """Interactive tool to help plan your portfolio and capstone project."""
    
    def __init__(self):
        """Initialize the portfolio planner."""
        self.skills = {}
        self.project_idea = {}
        self.technologies = []
        self.timeline = {}
    
    def assess_skills(self) -> Dict:
        """
        Assess skills across different areas.
        
        Returns:
            Dictionary with skill ratings
        """
        print("\n=== Skill Assessment ===")
        print("Rate yourself 1-5 (1=Beginner, 5=Expert)\n")
        
        categories = {
            "Data Engineering": [
                "Data formats (CSV, Parquet, Avro)",
                "Spark and distributed processing",
                "Kafka and streaming",
                "Data quality and testing"
            ],
            "Machine Learning": [
                "ML fundamentals (scikit-learn)",
                "Deep learning (PyTorch)",
                "MLOps and deployment",
                "Model optimization"
            ],
            "Generative AI": [
                "LLM fundamentals and prompting",
                "RAG systems",
                "AI agents",
                "Multi-modal AI"
            ],
            "APIs & Deployment": [
                "FastAPI development",
                "Testing and validation",
                "System integration",
                "Cloud deployment"
            ]
        }
        
        for category, skills in categories.items():
            print(f"\n{category}:")
            self.skills[category] = {}
            for skill in skills:
                rating = self._get_rating(f"  {skill}")
                self.skills[category][skill] = rating
        
        return self.skills
    
    def _get_rating(self, prompt: str) -> int:
        """Get rating from user (mock for automated testing)."""
        # In real usage, this would use input()
        # For testing, return a default value
        return 4
    
    def identify_strengths(self) -> List[str]:
        """
        Identify top strengths based on skill assessment.
        
        Returns:
            List of top skills
        """
        all_skills = []
        for category, skills in self.skills.items():
            for skill, rating in skills.items():
                all_skills.append((skill, rating, category))
        
        # Sort by rating
        all_skills.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 5
        top_skills = all_skills[:5]
        
        print("\n=== Your Top Strengths ===")
        for skill, rating, category in top_skills:
            print(f"- {skill} ({category}): {rating}/5")
        
        return [skill for skill, _, _ in top_skills]
    
    def suggest_project_focus(self) -> str:
        """
        Suggest project focus based on skills.
        
        Returns:
            Suggested focus area
        """
        # Calculate average rating per category
        category_avg = {}
        for category, skills in self.skills.items():
            avg = sum(skills.values()) / len(skills)
            category_avg[category] = avg
        
        # Find strongest category
        strongest = max(category_avg, key=category_avg.get)
        
        print(f"\n=== Suggested Focus ===")
        print(f"Based on your skills, consider focusing on: {strongest}")
        print(f"Average rating: {category_avg[strongest]:.1f}/5")
        
        return strongest
    
    def generate_project_ideas(self, focus: str) -> List[Dict]:
        """
        Generate project ideas based on focus area.
        
        Args:
            focus: Focus area (category name)
            
        Returns:
            List of project ideas
        """
        project_templates = {
            "Data Engineering": [
                {
                    "title": "Real-time Analytics Pipeline",
                    "description": "Build a streaming pipeline with Kafka and Spark",
                    "components": ["Kafka", "Spark Streaming", "Delta Lake", "FastAPI"],
                    "complexity": "Medium"
                },
                {
                    "title": "Data Quality Framework",
                    "description": "Automated data quality monitoring system",
                    "components": ["Great Expectations", "Airflow", "PostgreSQL", "Dashboard"],
                    "complexity": "Medium"
                },
                {
                    "title": "Lakehouse Platform",
                    "description": "Medallion architecture with Iceberg",
                    "components": ["Iceberg", "Spark", "dbt", "Superset"],
                    "complexity": "High"
                }
            ],
            "Machine Learning": [
                {
                    "title": "ML Model Deployment Platform",
                    "description": "End-to-end ML pipeline with monitoring",
                    "components": ["Scikit-learn", "MLflow", "FastAPI", "Docker"],
                    "complexity": "Medium"
                },
                {
                    "title": "Computer Vision Application",
                    "description": "Image classification with deployment",
                    "components": ["PyTorch", "Hugging Face", "FastAPI", "React"],
                    "complexity": "Medium"
                },
                {
                    "title": "AutoML System",
                    "description": "Automated model training and selection",
                    "components": ["Scikit-learn", "Optuna", "MLflow", "Streamlit"],
                    "complexity": "High"
                }
            ],
            "Generative AI": [
                {
                    "title": "Domain-Specific RAG System",
                    "description": "RAG for technical documentation",
                    "components": ["LangChain", "ChromaDB", "Ollama", "FastAPI"],
                    "complexity": "Medium"
                },
                {
                    "title": "AI Agent Platform",
                    "description": "Multi-tool AI agent with LangGraph",
                    "components": ["LangGraph", "OpenAI", "Tools", "Streamlit"],
                    "complexity": "High"
                },
                {
                    "title": "Multi-Modal Content Analyzer",
                    "description": "Analyze text, images, and audio",
                    "components": ["Transformers", "Whisper", "FastAPI", "React"],
                    "complexity": "High"
                }
            ],
            "APIs & Deployment": [
                {
                    "title": "Data API Platform",
                    "description": "Production-ready data API with caching",
                    "components": ["FastAPI", "Redis", "PostgreSQL", "Docker"],
                    "complexity": "Medium"
                },
                {
                    "title": "Microservices Architecture",
                    "description": "Multiple services with API gateway",
                    "components": ["FastAPI", "Docker", "Kubernetes", "Kong"],
                    "complexity": "High"
                },
                {
                    "title": "Serverless Data Pipeline",
                    "description": "AWS Lambda-based pipeline",
                    "components": ["Lambda", "S3", "DynamoDB", "API Gateway"],
                    "complexity": "Medium"
                }
            ]
        }
        
        ideas = project_templates.get(focus, [])
        
        print(f"\n=== Project Ideas for {focus} ===")
        for i, idea in enumerate(ideas, 1):
            print(f"\n{i}. {idea['title']}")
            print(f"   {idea['description']}")
            print(f"   Technologies: {', '.join(idea['components'])}")
            print(f"   Complexity: {idea['complexity']}")
        
        return ideas
    
    def create_project_plan(self, project: Dict) -> Dict:
        """
        Create detailed project plan.
        
        Args:
            project: Project idea dictionary
            
        Returns:
            Detailed project plan
        """
        plan = {
            "title": project["title"],
            "description": project["description"],
            "technologies": project["components"],
            "phases": {
                "Week 1: Foundation": [
                    "Set up repository structure",
                    "Define data sources and schemas",
                    "Create initial pipeline/model",
                    "Write unit tests"
                ],
                "Week 2: Core Implementation": [
                    "Build main features",
                    "Integrate components",
                    "Add error handling",
                    "Create API endpoints"
                ],
                "Week 3: Polish & Testing": [
                    "Comprehensive testing",
                    "Performance optimization",
                    "Documentation",
                    "Code review and refactoring"
                ],
                "Week 4: Deployment": [
                    "Deploy to cloud",
                    "Set up monitoring",
                    "Create demo/video",
                    "Write blog post"
                ]
            },
            "success_criteria": [
                "All tests passing",
                "Deployed and accessible",
                "Complete documentation",
                "Demo ready"
            ]
        }
        
        print(f"\n=== Project Plan: {plan['title']} ===")
        print(f"\nDescription: {plan['description']}")
        print(f"\nTechnologies: {', '.join(plan['technologies'])}")
        print("\nTimeline:")
        for phase, tasks in plan["phases"].items():
            print(f"\n{phase}:")
            for task in tasks:
                print(f"  - {task}")
        
        return plan
    
    def create_portfolio_structure(self) -> Dict:
        """
        Create portfolio repository structure.
        
        Returns:
            Portfolio structure dictionary
        """
        structure = {
            "README.md": "Portfolio overview and introduction",
            "projects/": {
                "01-capstone/": "Your main capstone project",
                "02-project/": "Second featured project",
                "03-project/": "Third featured project"
            },
            "mini-projects/": {
                "rag-system/": "RAG system from Day 84",
                "ai-agent/": "AI agent from Day 91",
                "integration/": "Integration project from Day 98"
            },
            "skills/": {
                "data-engineering.md": "Data engineering skills summary",
                "machine-learning.md": "ML skills summary",
                "generative-ai.md": "GenAI skills summary"
            },
            "docs/": {
                "resume.pdf": "Your resume",
                "certifications/": "Course certificates"
            }
        }
        
        print("\n=== Portfolio Structure ===")
        self._print_structure(structure)
        
        return structure
    
    def _print_structure(self, structure: Dict, indent: int = 0):
        """Print structure recursively."""
        for key, value in structure.items():
            print("  " * indent + f"├── {key}")
            if isinstance(value, dict):
                self._print_structure(value, indent + 1)
            else:
                print("  " * (indent + 1) + f"└── {value}")
    
    def save_plan(self, filename: str = "portfolio_plan.json"):
        """
        Save portfolio plan to file.
        
        Args:
            filename: Output filename
        """
        plan = {
            "skills": self.skills,
            "project_idea": self.project_idea,
            "technologies": self.technologies,
            "timeline": self.timeline
        }
        
        with open(filename, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"\n✅ Portfolio plan saved to {filename}")
    
    def run_interactive(self):
        """Run interactive portfolio planning session."""
        print("=" * 50)
        print("Portfolio Planning Tool")
        print("=" * 50)
        
        # Step 1: Assess skills
        self.assess_skills()
        
        # Step 2: Identify strengths
        strengths = self.identify_strengths()
        
        # Step 3: Suggest focus
        focus = self.suggest_project_focus()
        
        # Step 4: Generate ideas
        ideas = self.generate_project_ideas(focus)
        
        # Step 5: Create plan for first idea
        if ideas:
            self.project_idea = self.create_project_plan(ideas[0])
        
        # Step 6: Create portfolio structure
        self.create_portfolio_structure()
        
        # Step 7: Save plan
        self.save_plan()
        
        print("\n" + "=" * 50)
        print("Portfolio planning complete!")
        print("=" * 50)


if __name__ == "__main__":
    print("Day 99: Portfolio Planning Tool\n")
    
    planner = PortfolioPlanner()
    
    # Demo: Manual skill assessment
    print("=== Demo: Skill Assessment ===")
    planner.skills = {
        "Data Engineering": {
            "Data formats": 4,
            "Spark": 5,
            "Kafka": 4,
            "Data quality": 3
        },
        "Machine Learning": {
            "ML fundamentals": 4,
            "Deep learning": 3,
            "MLOps": 4,
            "Optimization": 3
        },
        "Generative AI": {
            "LLM fundamentals": 5,
            "RAG systems": 5,
            "AI agents": 4,
            "Multi-modal": 4
        },
        "APIs & Deployment": {
            "FastAPI": 5,
            "Testing": 4,
            "Integration": 4,
            "Cloud": 3
        }
    }
    
    # Identify strengths
    strengths = planner.identify_strengths()
    
    # Suggest focus
    focus = planner.suggest_project_focus()
    
    # Generate ideas
    ideas = planner.generate_project_ideas(focus)
    
    # Create plan
    if ideas:
        plan = planner.create_project_plan(ideas[0])
    
    # Create structure
    structure = planner.create_portfolio_structure()
    
    print("\n✅ Portfolio planning demo complete!")
