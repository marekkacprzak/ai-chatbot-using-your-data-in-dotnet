using ChatBot.Models;
using Microsoft.Extensions.AI;

namespace ChatBot.Services;

public class RagQuestionService(
    VectorSearchService vectorSearch,
    IChatClient client,
    ChatOptions chatOptions,
    PromptService promptService,
    VectorSearchServiceWithHyde vectorSearchWithHyde)
{
    public async Task<string> AnswerQuestion(string question)
    {
        // 1. Search using the original question (for baseline)
        var searchResultsOriginal = await vectorSearch.FindTopKChunks(question, 5);
        
        // 2. Search using the HYDE hypothesis
        var searchResultsHyde = await vectorSearchWithHyde.FindTopKChunks(question, 5);
        
        // 3. Combine results using Reciprocal Rank Fusion (RRF)
        var fusedResults = FuseSearchResults(searchResultsOriginal, searchResultsHyde);
        
        // 4. Create the user prompt with fused results
        var systemPrompt = promptService.RagSystemPrompt;

        var userPrompt = $@"User question:
{question}

Retrieved article sections:
{String.Join("\n\n", fusedResults.Select(chunk => @$"Title: {chunk.Title}
Section: {chunk.Section}
Part: {chunk.ChunkIndex + 1}
Content: {chunk.Content}
URL:{chunk.SourcePageUrl}"))}
";

        var messages = (new[] {
            new ChatMessage(ChatRole.System, systemPrompt),
            new ChatMessage(ChatRole.User, userPrompt)
        }).ToList();

        var response = await client.GetResponseAsync(messages, chatOptions);

        return response.Text;
    }
    
    private List<DocumentChunk> FuseSearchResults(List<DocumentChunk> original, List<DocumentChunk> hyde)
    {
        // Create SearchHit objects with actual scores for RRF
        var originalHits = original.Select(chunk => new SearchHit(chunk.Id!, chunk.Score)).ToList();
        var hydeHits = hyde.Select(chunk => new SearchHit(chunk.Id!, chunk.Score)).ToList();
        
        // Apply RRF fusion using actual scores
        var fusedHits = HydeFusion.ReciprocalRankFusion([originalHits, hydeHits], 5);
        
        // Map back to DocumentChunk objects
        var fusedChunkIds = fusedHits.Select(hit => hit.Id).ToHashSet();
        
        // Combine both result sets, prioritizing fused results
        var allChunks = original.Concat(hyde).ToList();
        return allChunks.Where(chunk => fusedChunkIds.Contains(chunk.Id!)).ToList();
    }
}