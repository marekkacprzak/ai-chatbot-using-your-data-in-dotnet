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
    public async Task<string> AnswerQuestionRrf(string question)
    {
        // 1. Search using the original question (for baseline)
        var searchResultsOriginal = await vectorSearch.FindTopKChunks(question, 15);
        
        // 2. Search using the HYDE hypothesis
        var searchResultsHyde = await vectorSearchWithHyde.FindTopKChunks(question, 15);
        
        // 3. Combine results using Reciprocal Rank Fusion (RRF)
        var fusedResults = FuseSearchResults(searchResultsOriginal, searchResultsHyde);
        
        return  await AnswerQuestion(question, fusedResults);
    }
    public async Task<string> AnswerQuestionMmr(string question)
    {
        // 1. Search using the original question (for baseline)
        var searchResultsOriginal = await vectorSearch.FindTopKChunks(question, 15);
        
        // 2. Search using the HYDE hypothesis
        var searchResultsHyde = await vectorSearchWithHyde.FindTopKChunks(question, 15);

        var concatScoreAndEmbedingDictionary = searchResultsOriginal
            .Concat(searchResultsHyde)
            .DistinctBy(a => a.Id)
            .ToDictionary(a => a.Id, a => new { a.Score, a.Embeding });
        
        var concatDocumentChunkDictionary = searchResultsOriginal
            .Concat(searchResultsHyde)
            .DistinctBy(a => a.Id)
            .ToDictionary(a => a.Id, a => a);

        var candidates = concatScoreAndEmbedingDictionary
            .Select(a=> new SearchHit(Id: a.Key, Score: a.Value.Score))
            .ToList();
        
        Func<string, string, double> docSim = (id1, id2) =>
            HydeFusion.CosineSimilarity(concatScoreAndEmbedingDictionary[id1].Embeding!, concatScoreAndEmbedingDictionary[id2].Embeding!);
        // Combine results using Maximal Marginal Relevance (MMR)
        var mmrResults = HydeFusion.MaximalMarginalRelevance(candidates, docSim, 0.7f, 15);
        var mmrContentResults = mmrResults.Select(a => concatDocumentChunkDictionary[a.Id]).ToList();
        return  await AnswerQuestion(question, mmrContentResults);
    }
    
    private async Task<string> AnswerQuestion(string question, List<DocumentChunk> fusedResults)
    {
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
        var fusedHits = HydeFusion.ReciprocalRankFusion([originalHits, hydeHits], 10);
        
        // Map back to DocumentChunk objects
        var fusedChunkIds = fusedHits.Select(hit => hit.Id).ToHashSet();
        
        // Combine both result sets, prioritizing fused results
        var allChunks = original.Concat(hyde).DistinctBy(a => a.Id).ToList();
        return allChunks.Where(chunk => fusedChunkIds.Contains(chunk.Id!)).ToList();
    }
}