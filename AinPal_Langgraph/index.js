const express = require('express');
const neo4j = require('neo4j-driver');
const { ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { OllamaEmbeddings } = require('@langchain/ollama');
require('dotenv').config();
const app = express();
app.use(express.json());
const port = 8000;
const neo4jDriver =  neo4j.driver('bolt://localhost:7999',neo4j.auth.basic(process.env.NEO4J_USERNAME, process.env.NEO4J_PASSWORD) 
);

const runQuery = async (query, params = {}) => {
    const session = neo4jDriver.session();
    try {
        const result = await session.run(query, params);
        return result.records;
    } finally {
        await session.close();
    }
};


app.post("/chat", async (req, res) => {
    const { queryText } = req.body;

    if (!queryText) {
        return res.status(400).json({ error: "Query text is required" });
    }

    try {
        // Initialize embeddings and model
        ollamaEmbedder = new OllamaEmbeddings({model: "bge-m3"});
        
        const model = new ChatGoogleGenerativeAI({
            model: "gemini-2.5-pro-preview-05-06",
            temperature: 0,
            apiKey: process.env.GOOGLE_API_KEY,
        });

        // Query Expansion Step 1: Generate multiple queries from original query
        const queryExpansionPrompt = `
            You are an expert legal assistant specializing in Bangladeshi law. 
            I have a legal question that I need help breaking down into multiple sub-questions:
            "${queryText}"
            
            Please generate 3 alternative questions that:
            1. Break down different aspects of the original question
            2. Focus specifically on relevant Bangladeshi legal concepts
            3. Use legal terminology that might appear in Bangladeshi legal texts
            4. Are worded differently but seek the same information
            
            Return only the list of questions, nothing else.
        `;
        
        const expandedQueriesResponse = await model.invoke(queryExpansionPrompt);
        const expandedQueries = expandedQueriesResponse.content
            .split('\n')
            .filter(q => q.trim().length > 0)
            .slice(0, 3); // Ensure we have maximum 3 queries
        
        // Add the original query to the expanded queries
        const allQueries = [queryText, ...expandedQueries];
        
        // Query Expansion Step 2: Generate hypothetical answers for each query
        const hypotheticalAnswers = [];
        for (const query of allQueries) {
            const hypotheticalAnswerPrompt = `
                You are a legal expert on Bangladeshi law. Create a brief hypothetical answer to the following legal question
                based on your knowledge of Bangladeshi legal texts and principles:
                "${query}"
                
                Provide a concise answer that might appear in Bangladeshi legal documents. Focus on legal terminology,
                principles, and references that would likely be found in actual Bangladeshi law texts.
                Keep your answer under 150 words and focus strictly on legal aspects.
            `;
            
            const hypotheticalAnswer = await model.invoke(hypotheticalAnswerPrompt);
            hypotheticalAnswers.push(hypotheticalAnswer.content);
        }
        
        // Query Expansion Step 3: Create a refined query based on hypothetical answers
        const combinedHypotheticalAnswers = hypotheticalAnswers.join("\n\n");
        const refinedQueryPrompt = `
            You are an expert in Bangladeshi law and legal information retrieval.
            Based on the following hypothetical answers to related legal questions:
            
            ${combinedHypotheticalAnswers}
            
            The original question was: "${queryText}"
            
            Create a detailed and comprehensive search query that will help retrieve the most relevant legal information from
            Bangladeshi legal texts. The query should:
            1. Include specific legal terminology from Bangladeshi law
            2. Incorporate key concepts from the hypothetical answers
            3. Be optimized for semantic search in a multilingual (English and Bangla) legal database
            4. Be between 50-100 words in length
            
            Your output should be only the refined search query, nothing else.
        `;
        
        const refinedQueryResponse = await model.invoke(refinedQueryPrompt);
        const expandedQuery = refinedQueryResponse.content.trim();
        
        console.log("Original query:", queryText);
        console.log("Expanded query:", expandedQuery);
        
        // Continue with the hybrid search using the expanded query instead of original queryText
        
        // Step 1: Vector search for semantic similarity using expanded query
        const queryVector = await ollamaEmbedder.embedQuery(expandedQuery);
        const vectorSearchQuery = `
            CALL db.index.vector.queryNodes('text_chunks', 8, $vector)
            YIELD node, score
            RETURN node, score
        `;
        
        const vectorResults = await runQuery(vectorSearchQuery, { vector: queryVector });
        

        const keywords = expandedQuery.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 3); 
            
        let keywordSearchQuery = '';
        let keywordParams = {};
        
        if (keywords.length > 0) {
            keywordSearchQuery = `
                MATCH (chunk:TextChunk)
                WHERE ${keywords.map((_, i) => `toLower(chunk.chunk_text) CONTAINS $keyword${i}`).join(' OR ')}
                RETURN chunk as node, 0.5 as score
                LIMIT 8
            `;
            
            keywords.forEach((keyword, i) => {
                keywordParams[`keyword${i}`] = keyword;
            });
        }
        

        const keywordResults = keywords.length > 0 ? 
            await runQuery(keywordSearchQuery, keywordParams) : [];
        

        const allResults = [...vectorResults, ...keywordResults];
        const uniqueNodes = new Map();
        
        allResults.forEach(record => {
            const node = record.get("node");
            const score = record.get("score");
            const nodeId = node.identity.toString();
            

            if (!uniqueNodes.has(nodeId) || uniqueNodes.get(nodeId).score < score) {
                uniqueNodes.set(nodeId, { node, score });
            }
        });
        

        const combinedResults = Array.from(uniqueNodes.values())
            .sort((a, b) => b.score - a.score)
            .slice(0, 10); // 
        

        const expandedNodes = new Set();
        const expandedResults = [];
        
        for (const result of combinedResults) {
            const nodeId = result.node.identity.toString();
            if (!expandedNodes.has(nodeId)) {
                expandedNodes.add(nodeId);
                expandedResults.push(result);
                
                // Fetch adjacent chunks
                const expandQuery = `
                    MATCH (node)-[:NEXT_CHUNK*1..2]->(next_chunk)
                    WHERE id(node) = $nodeId
                    RETURN next_chunk as node, 0.4 as score
                    LIMIT 3
                `;
                
                const expandedChunks = await runQuery(expandQuery, { nodeId: result.node.identity });
                
                for (const expandedRecord of expandedChunks) {
                    const expandedNode = expandedRecord.get("node");
                    const expandedNodeId = expandedNode.identity.toString();
                    
                    if (!expandedNodes.has(expandedNodeId)) {
                        expandedNodes.add(expandedNodeId);
                        expandedResults.push({ 
                            node: expandedNode, 
                            score: expandedRecord.get("score") 
                        });
                    }
                }
            }
        }
        

        const retrievedDocs = expandedResults
            .map((result) => {
                const node = result.node;
                const chunkText = node.properties.chunk_text || "";
                const lawTitle = node.properties.law_title || "";
                const sectionNumber = node.properties.section_number || "";
        
                return `\n\n[${lawTitle} | Section ${sectionNumber}]\n ${chunkText}`;
            })
            .join("\n");
        

        expandedResults.forEach((result) => {
            const node = result.node;
            const chunkText = node.properties.chunk_text || "";
            const lawTitle = node.properties.law_title || "";
            const sectionNumber = node.properties.section_number || "";
            const score = result.score;
        
            console.log(`[Score: ${score.toFixed(3)}] [${lawTitle} | Section ${sectionNumber}] ${chunkText}`);
            console.log("-------------------------------------")
        });

        const prompt = `You are a law assistant for question-answering tasks on Bangladeshi legislature.
        Use the following pieces of retrieved context to answer the question. Answer the questions in english.
        Don't start your answer with "Based on the retrieved context, ...". And if the question is a greeting or something irrelevant, answer accordingly.
        If you don't know the answer, just say that you don't know. Don't say something like "The retrieved context is not enough to answer the question."
        Question: "${queryText}"\n\nContext: "${retrievedDocs}\n\nAnswer:"
        Answer:`;
        
        let response;
        try {
            response = await model.invoke(prompt);
            console.log("Response:", response.content);
        }
        catch (error) {
            console.error("Error invoking model:", error);
            return res.status(500).json({ error: "An error occurred while processing the request" });
        }

        res.status(200).json({
            answer: response.content,
            debug: {
                retrievedDocCount: expandedResults.length,
                vectorResultsCount: vectorResults.length,
                keywordResultsCount: keywordResults.length,
                combinedResultsCount: combinedResults.length
            }
        });
    
    } catch (error) {
        console.error("Error handling /chat request:", error);
        res.status(500).json({ error: "An error occurred while processing the request" });
    }
});
   

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});