from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from dotenv import load_dotenv
import os

load_dotenv()

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

hf_medllm = LLM(model="huggingface/ProbeMedicalYonseiMAILab/medllama3-v20",
            base_url="https://api-inference.huggingface.co/models/ProbeMedicalYonseiMAILab/medllama3-v20",
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            provider="huggingface")

ol_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)


@CrewBase
class Hack4SyriaMedai:
    """Hack4SyriaMedai crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def translation_agent(self) -> Agent:
        print(f"Translation agent config: {self.agents_config['translation_agent']}")
        return Agent(config=self.agents_config["translation_agent"], verbose=True,llm= ol_llm )

    @agent
    def privacy_agent(self) -> Agent:
        return Agent(config=self.agents_config["privacy_agent"], verbose=True, llm= ol_llm )

    @agent
    def first_line_support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["first_line_support_agent"], verbose=True, llm= ol_llm # Loads from .env
)
    
    @agent
    def image_analyst_agent(self) -> Agent:
        return Agent(config=self.agents_config["image_analyst_agent"], verbose=True, llm= ol_llm, multimodal=True)
    @agent
    def diagnosis_agent(self) -> Agent:
        return Agent(config=self.agents_config["diagnosis_agent"], verbose=True, llm= ol_llm)

    @agent
    def cardiology_specialist_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["cardiology_specialist_agent"], verbose=True,llm= ol_llm
        )

    @agent
    def medical_knowledge_agent(self) -> Agent:
        return Agent(config=self.agents_config["medical_knowledge_agent"], verbose=True, llm= ol_llm)
    

    @agent
    def final_output_agent(self) -> Agent:
        return Agent(config=self.agents_config["final_output_agent"], verbose=True, llm= ol_llm)
    
    @agent
    def final_translator_agent(self) -> Agent:
        return Agent(config=self.agents_config["final_translator_agent"], verbose=True, llm= ol_llm)


    @task
    def translation_from_ar(self) -> Task:
        print(f"{self.tasks_config['translation_from_ar']}")
        return Task(
            config=self.tasks_config["translation_from_ar"],
            agent=self.translation_agent(),
            output_file="output/translated_text_ar.txt",
        )

    @task
    def privacy_check(self) -> Task:
        return Task(
            config=self.tasks_config["privacy_check"], agent=self.privacy_agent(),
            output_file="output/privacy_check_output.txt"
        )

    @task
    def information_gathering(self) -> Task:
        return Task(
            config=self.tasks_config["information_gathering"],
            agent=self.first_line_support_agent(),
            output_file="output/information_gathering_output.txt",
            # human_input=True
        )
    @task
    def image_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["image_analysis"], agent=self.diagnosis_agent(),
            output_file="output/image_analysis_output.txt",
        )
    @task
    def initial_diagnosis(self) -> Task:
        return Task(
            config=self.tasks_config["initial_diagnosis"], agent=self.diagnosis_agent(),
            output_file="output/initial_diagnosis_output.txt",
        )

    @task
    def cardiac_consultation(self) -> Task:
        return Task(
            config=self.tasks_config["cardiac_consultation"],
            agent=self.cardiology_specialist_agent(),
            output_file="output/cardiac_consultation_output.txt",
        )

    @task
    def medical_validation(self) -> Task:
        return Task(
            config=self.tasks_config["medical_validation"],
            agent=self.medical_knowledge_agent(),
            tools=[SerperDevTool()],
            output_file="output/medical_validation_output.txt",
        )

    @task
    def final_report_generation(self) -> Task:
        return Task(
            config=self.tasks_config["final_report_generation"],
            agent=self.final_output_agent(),
            output_file="output/final_report.txt",
        )

    @task
    def final_translation(self) -> Task:
        return Task(
            config=self.tasks_config["final_translation"],
            agent=self.translation_agent(),
            output_file="output/final_report_ar.txt",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Hack4SyriaMedai crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
