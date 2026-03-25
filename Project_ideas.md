1. Quizz erstellung durch Paper-input
    Benötigt wird:
        a. Datensatz
        b. RAG (Retrieval augmented reality)
    Datensätze: SQuAD, Wikipedia Dumps
    Tools: BM25 / FAISS
    Bonus Idee: ohne Retrieval vs. mit Retrieval

2. Bias oder Fairness in kleinen Languuge models (Wie fair/gender orientiert sind kleine Models)
    Benötigt wird:
        a. Model
        b. Ethische Basis
    Datensätze: CrowS-Pairs, WinoBias, StereoSet-artige Benschmarks,

3. Song text generation Programm
    Benötigt wird:
        a. Lama3b Model
        b. prompt window
    Datensätze: Genius Song Lyrics Dataset, Million Song Dataset, Kaggle Song Lyrics Datasets

4. Rezept finder/generator; Füge deine vorhandenen Zutaten ein und das Modell schlägt dir Rezepte vor.
    Benötigt wird: 
        a. Modell
            - Retrieval
            - LM-only
            - Retrieval + LM
        b. parameter descision - Wie/Welche Zutaten dürfen fehlen?
    Datensätze: RecipeNLG (cooking recipes dataset), Recipe1M, Food.com Recipes and Reviews Dataset, Epicurious Recipes Dataset

5. DnD NPC generator - Starting with character creation (through prompt input), ending with personalised conversations    based on Character
    Benötigt wird:
        a. json Datei pro Character
        b. Model Character creation
            - Self made?
        c. Model NPC-Conversation
            - lama3b
        Datensatz: Persona-Chat Dataset, DnD Character Database (Maluna), Gutenberg Project Corpus (fantasy sprache), Api Docs - Open5e (aktuelle Version)


