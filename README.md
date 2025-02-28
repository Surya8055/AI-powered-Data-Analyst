<h1>🎮 GenBI AI-powered Data Analyst 📊</h1>

<h2>📌 Project Overview</h2>  
<p>Welcome to my project that <strong>analyzes and visualizes video game sales data</strong> using <strong>data analysis</strong> and <strong>Large Language Models (LLMs)</strong>. The dataset, known as the <strong>Vgsales</strong> table, provides detailed information on video games, including 🎯 <strong>sales performance</strong>, 🎮 <strong>platforms</strong>, 📊 <strong>genres</strong>, ⭐ <strong>user ratings</strong>, and 👥 <strong>player counts</strong>.</p>  

<h2>🚀 Project Goal</h2>  
<p>The goal of this project is to develop an <strong>AI-powered Data Analyst Chatbot</strong> that can answer user queries by providing either <strong>text-based results</strong> or <strong>visualizations</strong> 📈. The project focuses on <strong>Generative AI techniques</strong>, including 🧠 <strong>Prompt Engineering</strong>, 🤖 <strong>integrating LLM APIs into real-world applications</strong>, 📊 <strong>data analysis</strong>, and 🏆 <strong>aspects of Machine Learning and NLP</strong>.</p>  

![image](https://github.com/user-attachments/assets/60a78f53-6af5-4a06-8251-13dac0948cbe)

<h1>🛠️ Libraries Used</h1>

<p>This project uses a combination of powerful Python libraries for data analysis, visualization, and NLP:</p>

<ul>
    <li><strong>Pandas</strong>: For efficient data manipulation and handling of the <code>Vgsales</code> dataset.</li>
    <li><strong>Matplotlib & Seaborn</strong>: For generating eye-catching visualizations like line charts and pie charts.</li>
    <li><strong>LangChain, OpenAI, and Transformers</strong>: To integrate <strong>Large Language Models (LLMs)</strong> for enhanced analysis through <strong>Natural Language Processing</strong>.</li>
    <li><strong>Torch</strong>: A deep learning library that supports LLMs for more advanced analysis.</li>
</ul>

<h1>📊 Dataset Overview</h1>

<p>The <strong>Vgsales</strong> dataset contains valuable insights into video game sales across different regions (e.g., North America, Europe), user ratings, player counts, and more. Key aspects of the dataset include:</p>

<table border="1" cellpadding="10">
    <tr>
        <th>Feature</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><strong>Game names</strong></td>
        <td>The name of the video game.</td>
    </tr>
    <tr>
        <td><strong>Platforms</strong></td>
        <td>The platform on which the game was released (e.g., PlayStation, Xbox).</td>
    </tr>
    <tr>
        <td><strong>Years of release</strong></td>
        <td>The year the game was released.</td>
    </tr>
    <tr>
        <td><strong>Genres</strong></td>
        <td>The genre of the game (e.g., Action, Adventure).</td>
    </tr>
    <tr>
        <td><strong>Sales</strong></td>
        <td>Sales data for the game in different regions.</td>
    </tr>
    <tr>
        <td><strong>User ratings</strong></td>
        <td>User ratings given to the game.</td>
    </tr>
    <tr>
        <td><strong>Player counts</strong></td>
        <td>The number of players who engaged with the game.</td>
    </tr>
</table>

<p>This data allows us to explore trends such as:</p>

<ul>
    <li><strong>Year-on-year sales trends</strong></li>
    <li><strong>Platform popularity</strong></li>
    <li><strong>Genre categorization</strong></li>
</ul>


<p>This data allows us to explore trends such as:</p>

<ul>
    <li><strong>Year-on-year sales trends</strong></li>
    <li><strong>Platform popularity</strong></li>
    <li><strong>Genre categorization</strong></li>
</ul>

<h1>🎯 Objectives</h1>

<p>The main goals of this project are:</p>

<ul>
    <li>🕹️ <strong>Establishing sales trends</strong> in video games over time.</li>
    <li>🎮 <strong>Comparing platform and genre performance</strong> based on sales.</li>
    <li>📈 <strong>Visualizing key findings</strong> using dynamic graphics like line charts and pie charts.</li>
</ul>

<h1>📈 Examples of Visualization</h1>

<ul>
    <li>Line charts to illustrate <strong>year-over-year sales trends</strong>.</li>
    <li>Pie charts to show the <strong>geographic breakdown of sales</strong> for best-selling games like Grand Theft Auto V.</li>
</ul>

<h1>🔍 Interactive Features</h1>

<p>The project utilizes libraries like <strong>Jupyter widgets</strong> to build interactive elements, allowing users to query the dataset and generate real-time visual responses. This integration of high-end libraries ensures that the analysis is both engaging and insightful.</p>

<h1>🔧 Installation Instructions</h1>

<p>Follow these steps to run the project on your local machine:</p>

<pre><code>
1. Clone the repository:
   git clone https://github.com/Surya8055/video-game-sales-analysis.git

2. Navigate to the project directory:
   cd video-game-sales-analysis

3. Install the required dependencies:
   pip install -r requirements.txt
</code></pre>

<h2>🛠️ Step 1: Installing the Necessary Libraries</h1>

<p>In this step, we install the required libraries to work with LLMs (Large Language Models) and build our data analysis chatbot. These libraries will help in managing datasets, generating insights, and running models efficiently. Below are the key installations:</p>

<table border="1" cellpadding="10">
    <tr>
        <th>Library</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>📦 <strong>langchain</strong></td>
        <td>The core framework for working with Large Language Models (LLMs). It enables seamless interaction between LLMs and external data sources like SQL and APIs.</td>
    </tr>
    <tr>
        <td>🌐 <strong>langchain-community</strong></td>
        <td>Install the community package that includes additional LLMs for broader functionality and experimentations with models from different providers.</td>
    </tr>
    <tr>
        <td>🤖 <strong>openai==0.28</strong></td>
        <td>The OpenAI API package (version 0.28) allows access to GPT models, such as GPT-3.5 and GPT-4, to power our chatbot with advanced language understanding.</td>
    </tr>
    <tr>
        <td>📚 <strong>transformers</strong></td>
        <td>Hugging Face's library that provides open-source models like LLaMA and Falcon for various NLP tasks, enabling access to other powerful LLMs.</td>
    </tr>
    <tr>
        <td>⚡ <strong>torch</strong></td>
        <td>PyTorch is a deep learning framework required to run models efficiently, especially for tasks involving large amounts of data and computation.</td>
    </tr>
    <tr>
        <td>🧠 <strong>langchain_openai</strong></td>
        <td>This package connects LangChain with OpenAI’s models, allowing us to leverage OpenAI’s powerful language models in a streamlined way within the LangChain ecosystem.</td>
    </tr>
</table>

<p>To install all these packages, run the following commands in your terminal or Jupyter notebook:</p>

<pre><code>
!pip install langchain  # Core framework for working with LLMs
!pip install langchain-community # Install the community package containing LLMs
!pip install openai==0.28  # OpenAI API package (version 0.28) for GPT models
!pip install transformers  # Hugging Face's library for open-source models like LLaMA and Falcon
!pip install torch  # PyTorch, required for running deep learning models efficiently
!pip install langchain_openai  # Connect LangChain with OpenAI models
</code></pre>

<p>Once these libraries are installed, you'll be all set to move to the next step of building your AI-powered data analysis chatbot! 🚀</p>


<h2><strong>Step 2: Importing All the Necessary Functions</strong></h1>
<p>This section describes the core Python libraries and code used for data analysis, visualizations, and integrating LangChain for interactive features. The libraries include:</p>
<ul>
    <li><strong>Data Analysis & Visualization</strong> with Pandas, Matplotlib, and Seaborn for generating various charts and visual representations.</li>
    <li><strong>LangChain</strong> for integrating AI models to process and analyze data with natural language queries.</li>
    <li><strong>SQL & Database Integration</strong> for loading and querying data with SQL-based solutions.</li>
    <li><strong>User Interface</strong> features built with Jupyter widgets to interactively explore data.</li>
</ul>

<p>For the full implementation, refer to the code sections in the project files.</p>

<h1><strong>🤖 AI-Powered Data Analysis Chatbot 📊</strong></h1>

<p>Welcome to the AI-powered chatbot project! 🚀 This chatbot helps users analyze datasets and generate visualizations using OpenAI's GPT-3.5-turbo model, LangChain, SQL, and Python libraries such as Pandas, Matplotlib, and Seaborn. 💻📈 It's a smart assistant for data-driven decision-making! 💡</p>

<h1><strong>🔧 Steps</strong></h1>

<h2><strong>Step 3: 🔑 Hardcoding OpenAI API Key in Google Colab</strong></h2>
<p>In this step, we hardcode the OpenAI API key to communicate with the GPT-3.5-turbo model. This lets the chatbot interact with the OpenAI API, but remember—keep the API key secure! 🔒</p>

<pre><code>
import openai

# Hardcode your OpenAI API key
openai.api_key = 'your-openai-api-key-here'
</code></pre>

<h2><strong>Step 4: 🧑‍💻 Giving Dataset Context to the LLM</strong></h2>
<p>A function is created to explain the structure of the dataset to the AI model. It generates descriptions of the dataset columns to help the model understand the data before performing any analysis or visualization. 🗂️</p>

<pre><code>
def explain_dataset_to_llm():
    return """
    The dataset contains the following columns:
    1. 'Name': Name of the game
    2. 'Platform': The platform where the game was released
    3. 'Year_of_Release': Year the game was released
    4. 'Genre': Genre of the game
    5. 'NA_Sales': Sales in North America (in millions)
    6. 'EU_Sales': Sales in Europe (in millions)
    7. 'JP_Sales': Sales in Japan (in millions)
    8. 'Other_Sales': Sales in other regions (in millions)
    9. 'Global_Sales': Total global sales (in millions)
    10. 'User_Score': User rating of the game
    11. 'Critic_Score': Critic score of the game
    """
</code></pre>

<h2><strong>Step 5: 🧩 Creating Custom Functions</strong></h2>

<h3><strong>5.1 📂 Load and Connect Data to SQL</strong></h3>
<p>Functions are created to load data into SQL databases and establish connections, allowing for smooth data interactions. 📥💾</p>

<pre><code>
import sqlite3
import pandas as pd

def load_data_to_sql(file_path, db_name):
    conn = sqlite3.connect(db_name)
    df = pd.read_csv(file_path)
    df.to_sql('vgsales', conn, if_exists='replace', index=False)
    conn.close()
</code></pre>

<h3><strong>5.2 🧠 Initialize SQL Agent</strong></h3>
<p>This function initializes an SQL agent to connect the language model (LLM) and SQL database, enabling the chatbot to execute queries. 🔍</p>

<pre><code>
from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_sql_agent
from langchain.llms import OpenAI

def initialize_sql_agent():
    llm = OpenAI(model="gpt-3.5-turbo")
    conn = sqlite3.connect("vgsales.db")
    sql_agent = create_sql_agent(llm, conn)
    return sql_agent
</code></pre>

<h3><strong>5.3 📝 Answer Queries</strong></h3>
<p>Functions handle user queries about the dataset, using the LLM to generate intelligent, context-aware responses based on the database. 💬</p>

<pre><code>
def get_answer_to_query(query, sql_agent):
    response = sql_agent.run(query)
    return response
</code></pre>

<h3><strong>5.4 📊 Visualization Generation</strong></h3>
<p>Another custom function generates dynamic visualizations (charts, graphs) based on user input and dataset analysis. 📈</p>

<pre><code>
import matplotlib.pyplot as plt

def generate_sales_visualization(df):
    plt.figure(figsize=(10,6))
    df.groupby('Genre')['Global_Sales'].sum().plot(kind='bar')
    plt.title('Global Sales by Genre')
    plt.ylabel('Global Sales (in millions)')
    plt.show()
</code></pre>

<h2 style="font-size: 24px;">Step 6: 🧑‍💼 Building the AI-Powered Chatbot</h3>

<h3 style="font-size: 22px;">6.1 📤 File Upload and Dataset Loading</h4>
<p>The user is prompted to upload a CSV file via the files.upload() function. 📤</p>
<p>The filename of the uploaded file is retrieved and placed in the <strong>file_name</strong> variable. The CSV file is read into a pandas DataFrame (<strong>cleansedVgsales</strong>) via <code>pd.read_csv()</code>. The DataFrame is held in <strong>df</strong> and is used for further processing and analysis. 📊</p>

<h3 style="font-size: 22px;">6.2 🔄 User Prompt and Select Output Type: Text or Visual</h4>
<p>In this step, the user provides a prompt or question regarding the dataset. The prompt could be anything related to the data, such as "What is the sales trend over the years?" or "Can you compare the sales by region?" 🤔</p>
<p>Once the user enters the prompt, they can choose the desired output type. They can select either "Text" or "Visual." 🎯</p>
<p>If the "Text" option is selected, the LLM will connect to the database, query the dataset, and provide a detailed response based on the prompt. The model will go through the dataset and generate an answer based on the user's question. 💬</p>
<p>If the user selects the "Visual" option, the LLM will generate Python code for a visualization based on the dataset context defined earlier. This code might be for creating charts like bar graphs, pie charts, or scatter plots. 🎨</p>
<p>Before running the generated code, some preprocessing is done to ensure the code is ready for execution. The code is cleaned up to remove any potential issues and formatted for smooth execution. The code is then saved as a `.py` file and executed using the <code>exec()</code> function. 📈</p>
<p>Finally, the chatbot generates and displays the visual output, providing users with a graphical representation of the data. 🌟</p>

<h3 style="font-size: 22px;">6.3 📝 Create Context Explanation for Visual Output</h4>
<p>The <code>context_setting(cleansedVgsales)</code> function is called to generate explanations for the columns in the data. This context will be used later by the chatbot when generating visualizations for the user. 🔍📊</p>

<h3 style="font-size: 22px;">6.4 💾 Load Data into SQL Database</h4>
<p>The <code>load_data_to_sql(df, db_path)</code> function is called to load the data into an SQL database. The data is inserted into a table named "Vgsales" and made ready to query. 💾</p>

<h3 style="font-size: 22px;">6.5 🌐 Create SQL Connection</h4>
<p>The <code>establish_sql_connection(db_path)</code> function is called to establish a connection to the SQL database. The function returns a <strong>SQLDatabase</strong> object (<strong>db</strong>) to access the database. 🌐🔗</p>

<h3 style="font-size: 22px;">6.6 🤖 Initialize the SQL Agent</h4>
<p>The <code>create_sql_agent_from_db(db, llm)</code> function is called to create an SQL agent that connects the database and the language model. The agent is able to execute SQL queries and parse replies based on the data schema. 🤖💬</p>

<h3 style="font-size: 22px;">6.7 🧩 Add Widgets for User Input</h4>
<p>A Text widget is created to allow users to enter their queries. 📝</p>
<p>A ToggleButtons widget is used to let the user choose the output type (Text or Visual). 🔘</p>
<p>A Button widget is created for submitting the query. 🖱️</p>

<h3 style="font-size: 22px;">6.8 📝 Define the Message Handling Function</h4>
<p>The <code>handle_message(change)</code> function is invoked when the "Send" button is pressed. It first checks whether the user input is an exit command (e.g., "exit" or "quit"). Otherwise, it runs the query and displays text or visual output based on the option. 📝🔄</p>

<h3 style="font-size: 22px;">6.9 🎨 Process Visual Output Generation</h4>
<p>When the user selects "Visual", a prompt is created with the context description and the user's question. The prompt is forwarded to the LLM, and it generates Python code for creating visualizations. 📈</p>
<p>The code is sanitized, and it is written to a file, which is subsequently executed within the Python environment to create the plot. The plot is displayed using <code>plt.show()</code>. 🎨📊</p>

<h3 style="font-size: 22px;">6.10 💬 Handle Text Output Generation</h4>
<p>If the user selects "Text", the chatbot forms a query to the database via the SQL agent. The result of the query is received as a text-based output. 🗣️📄</p>

<h3 style="font-size: 22px;">6.11 🧼 Clear User Input After Sending Message</h4>
<p>After the user's question is processed, the input field is cleared for the next question. ✨🧹</p>

<h3 style="font-size: 22px;">6.12 🖥️ Define the Widgets Display</h4>
<p>The widgets (input field, output type toggle, send button, and chat display) are arranged neatly for user interaction. 🖱️🖥️</p>

<h3 style="font-size: 22px;">6.13 🚀 Initialize the Chatbot</h4>
<p>Finally, the <code>start_chatbot()</code> function is called to initialize the chatbot. This function ties everything together, from the SQL database to the OpenAI API key and context for visual output. 🌟</p>

<h1>📹 Walk Through Video</h1>

▶️ [Watch Walkthrough Video](https://drive.google.com/file/d/1B8XMWzRyd3lVjGRQMnrNH7GVgx5GeeoQ/view?usp=sharing)



<h1>🛠️📊 Flow Chart</h1>

![ai_data_processing_flowchart (1)](https://github.com/user-attachments/assets/b37f77ca-aa2b-4c0a-94b2-10972bf94f56)


<h1>🎯 Challenges</h1>

<table border="1" cellpadding="10">
    <tr>
        <th>Challenge</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>❌ LLMs Can't Directly Generate Graphs</td>
        <td>LLMs generate text but can't directly generate graphs, with additional steps like generating Python code and executing it independently for graphical results.</td>
    </tr>
    <tr>
        <td>🧩 Defining the Context is Crucial</td>
        <td>Clearly defined context is essential for accurate results, as LLMs need properly defined data descriptions to interpret and respond correctly.</td>
    </tr>
    <tr>
        <td>💡 Prompt Engineering is Essential and Difficult</td>
        <td>It is hard to generate proper prompts because even minor wording differences can influence the LLM's ability to provide correct, actionable feedback.</td>
    </tr>
    <tr>
        <td>🔗 Complicated Process of Connecting and Building SQL Database</td>
        <td>Connecting and querying the SQL database was a complicated data preparation and connectivity process, which required knowledge of SQL and API integration.</td>
    </tr>
    <tr>
        <td>🧠 Depth of Difficulty with Other LLMs (Starcoder, Deepseek, Gemini)</td>
        <td>Alternatives like Starcoder and Gemini were tested but lacked the accuracy and reliability of OpenAI’s GPT models, necessitating their use for this project.</td>
    </tr>
    <tr>
        <td>🎨 Designing a UI Without Revealing Backend Complexity</td>
        <td>Designing a user-friendly UI that hides the backend’s complexity, such as data processing and code execution, posed significant design challenges.</td>
    </tr>
    <tr>
        <td>⏳ RAG (Retrieval-Augmented Generation) Takes Time</td>
        <td>RAG processes for retrieving and generating insights from large datasets were time-consuming, impacting response times and real-time interactivity.</td>
    </tr>
</table>

<h1 style="font-size: 28px;">🎉 <strong>Conclusion</strong></h2>
<p>This project successfully integrates <strong>AI, data analysis, and interactive visualizations</strong> into an intelligent chatbot. 🤖💬 Users can upload datasets, pose queries, and receive <strong>text-based answers or dynamic visual outputs</strong>, powered by <strong>OpenAI's language model</strong>, <strong>SQL databases</strong>, and advanced <strong>data visualization</strong> techniques.</p>

<p>By leveraging <strong>LangChain</strong>, <strong>Pandas</strong>, <strong>Matplotlib</strong>, and <strong>Seaborn</strong>, the chatbot offers a seamless, interactive experience for <strong>data-driven decision-making</strong>. Whether you're looking to explore trends, compare sales across platforms, or analyze user ratings, this tool provides an intuitive and powerful interface for data exploration. 📊📈</p>

<p>This project represents a leap forward in combining <strong>artificial intelligence</strong> with <strong>data analysis</strong> to simplify complex tasks, making data exploration accessible to everyone, regardless of technical expertise. 🌟</p>



<h1>🚀 Future Scope</h1>

<h2>🔍 Exploring and Applying RAG in New Forms</h2>
<p>We plan to enhance RAG to reduce latency and increase real-time capability, enhancing data retrieval and processing techniques for greater efficiency.</p>

<h2>💻 Creating an Advanced UI</h2>
<p>Future UI improvements will focus on creating a more interactive, intuitive interface, enabling greater user experience through enhanced filtering and real-time visualization.</p>

<h2>🤖 Bot Can Produce Both Text and Visual Responses</h2>
<p>Improving the ability of the bot to offer both text and image generation in an integrated way will enhance user experience with both text analysis and immediate graphical output.</p>

<h2>🔧 Improving the Accuracy of the Models</h2>
<p>Future enhancements will involve model training enhancement for higher accuracy, particularly for questions in certain datasets, using fine-tuning or the use of domain-specific knowledge.</p>

<h2>📚 Building an LLM Intended for This Task</h2>
<p>Making an LLM custom-trained for this use will augment context understanding and accuracy, with reduced reliance on general-purpose models.</p>

<h2>💬 Add Conversation Thread with Window and Buffer Memories for LLM</h2>
<p>Conversational memory by window and buffer approaches will facilitate the chatbot to recall old conversations, making dialogues more continuous and meaningful.</p>

<h2><strong>🔓 License & Attribution</strong></h2>

<p>This project is open-source and available for use, modification, and distribution under the terms of the MIT License. While you are free to use, modify, and share the code, proper credit must be given to the original author. This includes mentioning the author’s name and providing a link to the original project when redistributing or modifying the code.</p>

<p>Any derivative works should also include the same crediting and licensing terms. By using this project, you agree to provide appropriate attribution.</p>
