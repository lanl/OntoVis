# Interactive CLI - Chat with AI + Volume Tools

✅ **Created!** An interactive conversational interface where you chat with an AI that has access to all OntoVis agents as tools.

## Launch It

```bash
python examples/interactive_cli.py
```

## What You Get

A chat interface where you can:
- Ask questions in natural language
- The AI automatically uses the right tools
- Maintains conversation context
- Remembers what you've analyzed

### Available Tools (Auto-Selected by AI)

1. **analyze_volume** - Analyze datasets, get statistics, identify features
2. **render_volume** - Create 3D visualizations
3. **list_files** - Show available data files

## Example Usage

### Example 1: Discovery → Analysis → Visualization

```
You: What files do I have?

AI: [Uses list_files tool]
Available files in data:
- foot_256x256x256_uint8.raw (16.0 MB)

You: Analyze the foot dataset

AI: [Uses analyze_volume tool with dimensions automatically]
Volume Analysis Complete!

Statistics:
- Shape: (256, 256, 256)
- Value range: 0.0 to 255.0
- Mean: 13.8, Median: 0.0

[Full feature analysis...]

You: Show me the bones

AI: [Uses render_volume tool]
Rendering Complete!
Saved to: foot_256x256x256_uint8_render.png
Threshold: 120
Explanation: Using p95 threshold to capture dense bone structures...
```

### Example 2: Specific Requests

```
You: Analyze ../data/foot_256x256x256_uint8.raw and tell me what intensity range the bones are in

AI: [Analyzes data]
Based on the analysis, bones are in the high-intensity range from approximately 100-255, with dense cortical bone at 180-255.

You: Render structures above 150

AI: [Creates visualization]
Done! Saved to foot_256x256x256_uint8_render.png
This shows only the densest bone structures.
```

### Example 3: Iterative Exploration

```
You: List my files in ../data

AI: Available files in ../data:
- foot_256x256x256_uint8.raw (16.0 MB)

You: What's in that file?

AI: [Analyzes it]
It's a CT scan of a human foot with bones, soft tissue, and air spaces.

You: Show me a visualization

AI: [Renders it]
Created a bone visualization at threshold 120.

You: Now show it at threshold 50 instead

AI: [Re-renders with different parameters]
Done! This shows more soft tissue structures.
```

## How It Works

The AI assistant:
1. **Understands your intent** from natural language
2. **Chooses the right tool(s)** automatically
3. **Fills in parameters** based on context
4. **Remembers previous interactions** (e.g., which file you analyzed)
5. **Returns results** in conversational format

## Tips

### For .raw files, mention dimensions once:
```
You: Analyze data/volume.raw (it's 256x256x256 uint8)
```

### Or let the AI ask:
```
You: Analyze data/volume.raw

AI: I need the dimensions for .raw files. What are they?

You: 256 by 256 by 256, uint8
```

### Chain requests naturally:
```
You: Analyze the foot data then show me the bones
```

The AI will do both in sequence!

### Ask follow-up questions:
```
You: What was the mean intensity?

You: What threshold did you use for rendering?

You: Can you render it darker?
```

## Commands

- Type anything in natural language
- Type `quit`, `exit`, or `bye` to leave
- Press Ctrl+C to interrupt

## Under the Hood

Built with:
- **LangGraph** for agent orchestration
- **LangChain** tool calling
- **Claude** as the LLM
- **OntoVis agents** as tools

The AI has full access to:
- `VolumeAnalysisAgent`
- `VolumeRenderAgent`
- File system operations

## Troubleshooting

**"Error initializing agent"**
- Check your `.config` file has valid API credentials

**"Error: .raw files require dimensions"**
- Specify dimensions in your message: "it's 256x256x256"

**Tool not being called**
- Be explicit: "Use the analyze tool on..." or "Render the..."

## Next Steps

Try it out! The AI will guide you through the process.

```bash
python examples/interactive_cli.py
```

Start with: `"What can you do?"` or `"What files do I have?"`
