using Microsoft.Extensions.AI;
using Pinecone;
using ChatBot.Services;
using OpenAI;
using Index = Pinecone.Index;
namespace ChatBot;

static class Startup
{
    public static async Task ConfigureServices(WebApplicationBuilder builder)
    {
        builder.Services.AddCors(options =>
        {
            options.AddPolicy("FrontendCors", policy =>
                policy
                    .WithOrigins("http://localhost:3000")
                    .AllowAnyHeader()
                    .AllowAnyMethod()
            );
        });

        var openAiClient = builder.GetLocalAiClient();

        await builder.AddPineconeIndex();
    
        builder.Services.AddSingleton<StringEmbeddingGenerator>(s => 
            openAiClient
            .GetEmbeddingClient(builder.RequireEnv("LLMEmbedding"))
            .AsIEmbeddingGenerator());

        builder.Services.AddSingleton<DocumentChunkStore>(s => new DocumentChunkStore());

        builder.Services.AddSingleton<VectorSearchService>();

        builder.Services.AddLogging(logging => logging.AddConsole().SetMinimumLevel(LogLevel.Information));

        builder.Services.AddSingleton<ILoggerFactory>(sp =>
            LoggerFactory.Create(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Information)));

        builder.Services.AddSingleton<IChatClient>(sp =>
         {
             var loggerFactory = sp.GetRequiredService<ILoggerFactory>();

             var client = openAiClient
                 .GetChatClient(builder.RequireEnv("LLMModel"))
                 .AsIChatClient();

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

        builder.Services.AddSingleton<WikipediaClient>();
        builder.Services.AddSingleton<IndexBuilder>();
        builder.Services.AddSingleton<RagQuestionService>();
        builder.Services.AddSingleton<ArticleSplitter>();
        builder.Services.AddSingleton<PromptService>();
    }

    extension(WebApplicationBuilder builder)
    {
        private async Task AddPineconeIndex()
        {
            var pineconeKey = builder.RequireEnv("PINECONE_API_KEY");
            var pinecone = new PineconeClient(pineconeKey);
            var indexName = "landmark-chunks-2";


            if (!((await pinecone.ListIndexesAsync()).Indexes?.Any(i => i.Name == indexName) ?? false))
            {
                // 2. Create a dense serverless index matching your vector dimensions
                await pinecone.CreateIndexAsync(new CreateIndexRequest
                {
                    Name = indexName,
                    Dimension = 3584, // Must match the length of your float[] Values
                    Metric = MetricType.Cosine,
                    Spec = new ServerlessIndexSpec
                    {
                        Serverless = new ServerlessSpec
                        {
                            Cloud = ServerlessSpecCloud.Aws,
                            Region = "us-east-1"
                        }
                    }
                });
                Index indexInfo;
                do
                {
                    await Task.Delay(2000);
                    indexInfo = await pinecone.DescribeIndexAsync(indexName);
                } while (!indexInfo.Status.Ready);
            }

            builder.Services.AddSingleton<IndexClient>(s => pinecone.Index(indexName));
        }

        private OpenAIClient GetLocalAiClient()
        {
            var lmstudioApiUrl = builder.RequireEnv("LMSTUDIO_API_URL");
            var localClientOptions = new OpenAIClientOptions
            {
                Endpoint = new Uri(lmstudioApiUrl)
            };

            var openAiClient = new OpenAIClient(
                new System.ClientModel.ApiKeyCredential("lmstudio"),
                localClientOptions
            );
            return openAiClient;
        }
    }
}
