"""
Main application module
"""
import asyncio
from src.core.researcher import DeepResearcher
from src.utils.config import settings
from src.ui.console import (
    display_welcome,
    get_user_query,
    get_search_params,
    display_results,
    ask_continue,
    display_progress
)

async def main():
    """Main function"""
    # Inicializa pesquisador
    researcher = DeepResearcher()
    
    # Interface principal
    display_welcome()
    
    while True:
        # Obtém query do usuário
        query = get_user_query()
        if query.lower() in ["sair", "exit", "quit"]:
            break
        
        # Obtém parâmetros
        params = get_search_params()
        
        # Realiza pesquisa
        with display_progress("Searching..."):
            results = await researcher.research(
                query=query,
                depth=params["depth"],
                breadth=params["breadth"]
            )
        
        # Exibe resultados
        display_results(results)
        
        # Pergunta se deseja continuar
        if not ask_continue():
            break
    
    print("\nThank you for using Deep Research Local!")

if __name__ == "__main__":
    asyncio.run(main()) 