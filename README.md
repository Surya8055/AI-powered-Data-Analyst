<h2>📊 Video Game Sales Analysis - Flowchart</h2>
<p>This flowchart illustrates the end-to-end process of analyzing video game sales data, from data uploading to AI-powered insights.</p>

```mermaid
graph TD;
    Start[🔘 <b style="font-size:22px">START</b>] -->|<b style="font-size:22px">Upload Data</b>| Upload[📂 <b style="font-size:22px">DATA UPLOAD</b>]
    Upload -->|<b style="font-size:22px">Clean & Preprocess</b>| Cleaning[🛠 <b style="font-size:22px">DATA CLEANING</b>]
    Cleaning -->|<b style="font-size:22px">Explore Data</b>| Exploration[📊 <b style="font-size:22px">DATA EXPLORATION</b>]
    
    Exploration -->|<b style="font-size:22px">Visualize Trends</b>| Visualization[📈 <b style="font-size:22px">DATA VISUALIZATION</b>]
    Visualization -->|<b style="font-size:22px">AI Analysis</b>| AI[🤖 <b style="font-size:22px">AI-POWERED QUERYING</b>]
    AI -->|<b style="font-size:22px">Generate Insights</b>| Results[📌 <b style="font-size:22px">RESULTS & INSIGHTS</b>]
    Results --> End[🔘 <b style="font-size:22px">END</b>]

    %% Adjust layout for two horizontal lines
    Cleaning -->|⬇ Move to second line ⬇| Visualization

    %% Enhanced styling
    style Start fill:#C0C0C0,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
    style Upload fill:#87CEEB,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
    style Cleaning fill:#FFD700,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
    style Exploration fill:#90EE90,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
    style Visualization fill:#FF6347,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
    style AI fill:#DA70D6,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
    style Results fill:#778899,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
    style End fill:#C0C0C0,stroke:#333,stroke-width:4px,font-size:22px,font-weight:bold
