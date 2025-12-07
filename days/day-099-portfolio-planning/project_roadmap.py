"""
Day 99: Portfolio Planning - Project Roadmap Generator
"""
from datetime import datetime, timedelta
from typing import Dict, List


class ProjectRoadmap:
    """Generate detailed project roadmap with tasks and milestones."""
    
    def __init__(self, project_name: str, duration_weeks: int = 4):
        """
        Initialize roadmap generator.
        
        Args:
            project_name: Name of the project
            duration_weeks: Project duration in weeks
        """
        self.project_name = project_name
        self.duration_weeks = duration_weeks
        self.tasks = []
        self.milestones = []
        self.start_date = datetime.now()
    
    def add_task(self, name: str, week: int, duration_days: int, dependencies: List[str] = None):
        """
        Add task to roadmap.
        
        Args:
            name: Task name
            week: Week number (1-based)
            duration_days: Task duration in days
            dependencies: List of task names this depends on
        """
        task = {
            "name": name,
            "week": week,
            "duration_days": duration_days,
            "dependencies": dependencies or [],
            "status": "pending"
        }
        self.tasks.append(task)
    
    def add_milestone(self, name: str, week: int, criteria: List[str]):
        """
        Add milestone to roadmap.
        
        Args:
            name: Milestone name
            week: Week number (1-based)
            criteria: Success criteria
        """
        milestone = {
            "name": name,
            "week": week,
            "criteria": criteria
        }
        self.milestones.append(milestone)
    
    def generate_default_roadmap(self, project_type: str):
        """
        Generate default roadmap based on project type.
        
        Args:
            project_type: Type of project (data_engineering, ml, genai, api)
        """
        if project_type == "data_engineering":
            self._generate_data_engineering_roadmap()
        elif project_type == "ml":
            self._generate_ml_roadmap()
        elif project_type == "genai":
            self._generate_genai_roadmap()
        elif project_type == "api":
            self._generate_api_roadmap()
        else:
            self._generate_generic_roadmap()
    
    def _generate_data_engineering_roadmap(self):
        """Generate roadmap for data engineering project."""
        # Week 1: Foundation
        self.add_task("Set up repository and CI/CD", 1, 1)
        self.add_task("Define data schemas", 1, 2)
        self.add_task("Set up Kafka/Spark environment", 1, 2, ["Set up repository and CI/CD"])
        self.add_task("Create data ingestion pipeline", 1, 3, ["Define data schemas"])
        self.add_milestone("Foundation Complete", 1, [
            "Repository set up with CI/CD",
            "Data schemas defined",
            "Basic pipeline running"
        ])
        
        # Week 2: Core Implementation
        self.add_task("Implement data transformations", 2, 3, ["Create data ingestion pipeline"])
        self.add_task("Add data quality checks", 2, 2, ["Implement data transformations"])
        self.add_task("Create Delta/Iceberg tables", 2, 2, ["Implement data transformations"])
        self.add_task("Build API endpoints", 2, 2)
        self.add_milestone("Core Features Complete", 2, [
            "Transformations working",
            "Quality checks in place",
            "API accessible"
        ])
        
        # Week 3: Polish & Testing
        self.add_task("Write comprehensive tests", 3, 3)
        self.add_task("Add monitoring and logging", 3, 2)
        self.add_task("Performance optimization", 3, 2, ["Write comprehensive tests"])
        self.add_task("Documentation", 3, 2)
        self.add_milestone("Testing Complete", 3, [
            "Test coverage > 80%",
            "Monitoring in place",
            "Documentation complete"
        ])
        
        # Week 4: Deployment
        self.add_task("Deploy to cloud", 4, 2)
        self.add_task("Set up production monitoring", 4, 1, ["Deploy to cloud"])
        self.add_task("Create demo and video", 4, 2)
        self.add_task("Write blog post", 4, 2)
        self.add_milestone("Project Complete", 4, [
            "Deployed and running",
            "Demo ready",
            "Blog post published"
        ])
    
    def _generate_ml_roadmap(self):
        """Generate roadmap for ML project."""
        # Week 1
        self.add_task("Data collection and exploration", 1, 3)
        self.add_task("Feature engineering", 1, 3, ["Data collection and exploration"])
        self.add_task("Set up MLflow", 1, 1)
        self.add_milestone("Data Ready", 1, ["Dataset prepared", "Features engineered"])
        
        # Week 2
        self.add_task("Model training and evaluation", 2, 3, ["Feature engineering"])
        self.add_task("Hyperparameter tuning", 2, 2, ["Model training and evaluation"])
        self.add_task("Model selection", 2, 2, ["Hyperparameter tuning"])
        self.add_milestone("Model Trained", 2, ["Best model selected", "Metrics documented"])
        
        # Week 3
        self.add_task("Create inference API", 3, 3, ["Model selection"])
        self.add_task("Add model monitoring", 3, 2, ["Create inference API"])
        self.add_task("Write tests", 3, 2)
        self.add_milestone("API Ready", 3, ["API working", "Tests passing"])
        
        # Week 4
        self.add_task("Deploy model", 4, 2, ["Create inference API"])
        self.add_task("Create demo", 4, 2)
        self.add_task("Documentation", 4, 2)
        self.add_milestone("Deployed", 4, ["Model in production", "Demo ready"])
    
    def _generate_genai_roadmap(self):
        """Generate roadmap for GenAI project."""
        # Week 1
        self.add_task("Set up LLM environment", 1, 1)
        self.add_task("Collect and process documents", 1, 3)
        self.add_task("Implement chunking strategy", 1, 2, ["Collect and process documents"])
        self.add_task("Set up vector database", 1, 2)
        self.add_milestone("Data Indexed", 1, ["Documents processed", "Vectors stored"])
        
        # Week 2
        self.add_task("Implement retrieval", 2, 2, ["Set up vector database"])
        self.add_task("Build generation pipeline", 2, 3, ["Implement retrieval"])
        self.add_task("Add prompt templates", 2, 2, ["Build generation pipeline"])
        self.add_milestone("RAG Working", 2, ["Retrieval accurate", "Generation quality good"])
        
        # Week 3
        self.add_task("Create API/UI", 3, 3)
        self.add_task("Add evaluation metrics", 3, 2)
        self.add_task("Optimize performance", 3, 2, ["Add evaluation metrics"])
        self.add_milestone("System Complete", 3, ["API working", "Performance optimized"])
        
        # Week 4
        self.add_task("Deploy system", 4, 2)
        self.add_task("Create demo", 4, 2)
        self.add_task("Write documentation", 4, 2)
        self.add_milestone("Deployed", 4, ["System live", "Demo ready"])
    
    def _generate_api_roadmap(self):
        """Generate roadmap for API project."""
        # Week 1
        self.add_task("Design API endpoints", 1, 2)
        self.add_task("Set up FastAPI project", 1, 1)
        self.add_task("Implement core endpoints", 1, 3, ["Design API endpoints"])
        self.add_task("Add Pydantic models", 1, 2, ["Implement core endpoints"])
        self.add_milestone("API Foundation", 1, ["Core endpoints working"])
        
        # Week 2
        self.add_task("Add authentication", 2, 2)
        self.add_task("Implement caching", 2, 2)
        self.add_task("Add rate limiting", 2, 2)
        self.add_task("Database integration", 2, 3)
        self.add_milestone("Features Complete", 2, ["Auth working", "Caching in place"])
        
        # Week 3
        self.add_task("Write tests", 3, 3)
        self.add_task("Add monitoring", 3, 2)
        self.add_task("Documentation", 3, 2)
        self.add_milestone("Testing Done", 3, ["Tests passing", "Docs complete"])
        
        # Week 4
        self.add_task("Deploy to cloud", 4, 2)
        self.add_task("Set up CI/CD", 4, 2)
        self.add_task("Create demo", 4, 2)
        self.add_milestone("Production Ready", 4, ["Deployed", "CI/CD working"])
    
    def _generate_generic_roadmap(self):
        """Generate generic roadmap."""
        for week in range(1, self.duration_weeks + 1):
            self.add_task(f"Week {week} tasks", week, 5)
            self.add_milestone(f"Week {week} milestone", week, ["Tasks complete"])
    
    def print_roadmap(self):
        """Print formatted roadmap."""
        print(f"\n{'=' * 60}")
        print(f"Project Roadmap: {self.project_name}")
        print(f"Duration: {self.duration_weeks} weeks")
        print(f"Start Date: {self.start_date.strftime('%Y-%m-%d')}")
        print(f"{'=' * 60}\n")
        
        for week in range(1, self.duration_weeks + 1):
            week_start = self.start_date + timedelta(weeks=week-1)
            week_end = week_start + timedelta(days=6)
            
            print(f"Week {week}: {week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}")
            print("-" * 60)
            
            # Print tasks for this week
            week_tasks = [t for t in self.tasks if t["week"] == week]
            if week_tasks:
                print("\nTasks:")
                for task in week_tasks:
                    deps = f" (depends on: {', '.join(task['dependencies'])})" if task['dependencies'] else ""
                    print(f"  • {task['name']} ({task['duration_days']} days){deps}")
            
            # Print milestones for this week
            week_milestones = [m for m in self.milestones if m["week"] == week]
            if week_milestones:
                print("\nMilestones:")
                for milestone in week_milestones:
                    print(f"  🎯 {milestone['name']}")
                    for criterion in milestone['criteria']:
                        print(f"     ✓ {criterion}")
            
            print()
    
    def export_markdown(self, filename: str = "roadmap.md"):
        """
        Export roadmap to markdown file.
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write(f"# {self.project_name} - Project Roadmap\n\n")
            f.write(f"**Duration**: {self.duration_weeks} weeks\n")
            f.write(f"**Start Date**: {self.start_date.strftime('%Y-%m-%d')}\n\n")
            
            for week in range(1, self.duration_weeks + 1):
                week_start = self.start_date + timedelta(weeks=week-1)
                week_end = week_start + timedelta(days=6)
                
                f.write(f"## Week {week}: {week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}\n\n")
                
                # Tasks
                week_tasks = [t for t in self.tasks if t["week"] == week]
                if week_tasks:
                    f.write("### Tasks\n\n")
                    for task in week_tasks:
                        deps = f" *(depends on: {', '.join(task['dependencies'])})*" if task['dependencies'] else ""
                        f.write(f"- [ ] {task['name']} ({task['duration_days']} days){deps}\n")
                    f.write("\n")
                
                # Milestones
                week_milestones = [m for m in self.milestones if m["week"] == week]
                if week_milestones:
                    f.write("### Milestones\n\n")
                    for milestone in week_milestones:
                        f.write(f"**{milestone['name']}**\n")
                        for criterion in milestone['criteria']:
                            f.write(f"- {criterion}\n")
                        f.write("\n")
        
        print(f"✅ Roadmap exported to {filename}")


if __name__ == "__main__":
    print("Day 99: Project Roadmap Generator\n")
    
    # Demo: Data Engineering Project
    print("=== Data Engineering Project Roadmap ===")
    roadmap = ProjectRoadmap("Real-time Analytics Pipeline", duration_weeks=4)
    roadmap.generate_default_roadmap("data_engineering")
    roadmap.print_roadmap()
    
    # Demo: ML Project
    print("\n" + "=" * 60)
    print("=== Machine Learning Project Roadmap ===")
    ml_roadmap = ProjectRoadmap("ML Model Deployment", duration_weeks=4)
    ml_roadmap.generate_default_roadmap("ml")
    ml_roadmap.print_roadmap()
    
    # Demo: GenAI Project
    print("\n" + "=" * 60)
    print("=== GenAI Project Roadmap ===")
    genai_roadmap = ProjectRoadmap("RAG System", duration_weeks=4)
    genai_roadmap.generate_default_roadmap("genai")
    genai_roadmap.print_roadmap()
    
    print("\n✅ Roadmap generation demo complete!")
