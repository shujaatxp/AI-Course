using LangChain;
using LangChain.Chains; // <-- Add this using directive
using LangChain.Chains.LLM;
using LangChain.Providers;
using LangChain.Providers.OpenAI;
using Microsoft.ML;

public class RealEstateAgent
{
    private readonly OpenAiChatModel _llm;
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<HouseData, HousePrediction> _predictor;

    public RealEstateAgent(string apiKey)
    {
        // Initialize OpenAI LLM
        _llm = new OpenAiChatModel(apiKey, "gpt-4o-mini");

        // Load ML.NET model
        _mlContext = new MLContext();
        var model = _mlContext.Model.Load("model.zip", out _);
        _predictor = _mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
    }

    private string PredictHousePrice(float size, float bedrooms)
    {
        var prediction = _predictor.Predict(new HouseData
        {
            Size = size,
            Bedrooms = bedrooms
        });
        return $"Predicted price: €{prediction.Price:F0}";
    }

    public async Task RunAsync()
    {
        Console.WriteLine("🏡 Real Estate AI Agent Ready! Type 'exit' to quit.\n");

        while (true)
        {
            Console.Write("You: ");
            var input = Console.ReadLine();
            if (input?.ToLower() == "exit") break;

            if (input != null && input.ToLower().Contains("predict"))
            {
                var numbers = input.Split(' ')
                                   .Where(s => float.TryParse(s, out _))
                                   .Select(float.Parse)
                                   .ToArray();

                if (numbers.Length >= 2)
                {
                    var result = PredictHousePrice(numbers[0], numbers[1]);
                    Console.WriteLine($"Agent: {result}");
                    continue;
                }
            }

            // LLM response
            var prompt = $"You are a helpful assistant. Answer the user: {input}";
            var chatRequest = ChatRequest.ToChatRequest(prompt);
            await foreach (var chatResponse in _llm.GenerateAsync(chatRequest))
            {
                Console.WriteLine($"Agent: {chatResponse.LastMessageContent}");
                break; // Only print the first response
            }
        }
    }
}