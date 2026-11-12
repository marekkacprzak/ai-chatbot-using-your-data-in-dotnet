using System;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using Pinecone;
using ChatBot.Services;
using OpenAI;

namespace ChatBot;

static class Startup
{
    public static void ConfigureServices(WebApplicationBuilder builder)
    {
        var pineconeKey = builder.RequireEnv("PINECONE_API_KEY");

        builder.Services.AddCors(options =>
        {
            options.AddPolicy("FrontendCors", policy =>
                policy
                    .WithOrigins("http://localhost:3000")
                    .AllowAnyHeader()
                    .AllowAnyMethod()
            );
        });
        var clientOptions = new OpenAIClientOptions
        {
            Endpoint = new Uri("http://192.168.8.131:1234/v1")
        };

        var openAIClient = new OpenAIClient(
            new System.ClientModel.ApiKeyCredential("lmstudio"),
            clientOptions
        );
        builder.Services.AddSingleton<StringEmbeddingGenerator>(s => openAIClient
            .GetEmbeddingClient("text-embedding-3-small")
            .AsIEmbeddingGenerator());

        builder.Services.AddSingleton<IndexClient>(s => new PineconeClient(pineconeKey).Index("youtube-transcripts"));

        builder.Services.AddSingleton<DocumentChunkStore>(s => new DocumentChunkStore());

        builder.Services.AddSingleton<VectorSearchService>();

        builder.Services.AddLogging(logging => logging.AddConsole().SetMinimumLevel(LogLevel.Information));

        builder.Services.AddSingleton<ILoggerFactory>(sp =>
            LoggerFactory.Create(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Information)));

        builder.Services.AddSingleton<IChatClient>(sp =>
         {
             var loggerFactory = sp.GetRequiredService<ILoggerFactory>();
             var client = new OpenAI.Chat.ChatClient(
                  "gpt-5-mini",
                  openAiKey).AsIChatClient();

             return new ChatClientBuilder(client)
                 .UseLogging(loggerFactory)
                 .UseFunctionInvocation(loggerFactory, c =>
                 {
                     c.IncludeDetailedErrors = true;
                 })
                 .Build(sp);
         });

        builder.Services.AddTransient<ChatOptions>(sp => new ChatOptions
        {
            Tools = FunctionRegistry.GetTools(sp).ToList(),
        });

        builder.Services.AddSingleton<IndexBuilder>();
        builder.Services.AddSingleton<RagQuestionService>();
        builder.Services.AddSingleton<TranscriptSplitter>();
        builder.Services.AddSingleton<PromptService>();

    }
}
