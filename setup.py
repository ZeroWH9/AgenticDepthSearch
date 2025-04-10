from setuptools import setup, find_packages

setup(
    name="deep-research-local",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain==0.3.23",
        "langchain-core>=0.3.51",
        "langchain-community>=0.3.21",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "rich>=13.7.0",
        "python-dotenv>=1.0.0",
        "tavily-python>=0.2.0",
        "chromadb>=0.4.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "asyncio>=3.4.3",
        "typing-extensions>=4.8.0",
        "langsmith>=0.0.65"
    ],
    python_requires=">=3.9",
) 