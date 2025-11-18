using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using OllamaSharp;
using OllamaSharp.Models.Chat;
using OllamaChatRole = OllamaSharp.Models.Chat.ChatRole;

class Program
{
    // ML.NET input/output classes
    public class HouseData
    {
        [LoadColumn(0)]
        public float Size { get; set; }

        [LoadColumn(1)]
        public float Bedrooms { get; set; }

        [LoadColumn(2)]
        public float Age { get; set; }

        [LoadColumn(3)]
        public float Price { get; set; } // Label
    }

    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }

    static async Task Main()
    {
        var mlContext = new MLContext(seed: 0);
        ITransformer model;
        IDataView dataView;

        // 1️⃣ Load CSV
        try
        {
            dataView = mlContext.Data.LoadFromTextFile<HouseData>("C:\\Personal\\AICourse\\MLNet\\MLOLLAMA\\house_data.csv", hasHeader: true, separatorChar: ',');
        }
        catch
        {
            Console.WriteLine("❌ Could not find house_data.csv");
            return;
        }

        // 2️⃣ Train or load model
        try
        {
            model = mlContext.Model.Load("HousePriceModel.zip", out _);
            Console.WriteLine("✅ Loaded existing model from HousePriceModel.zip");
        }
        catch
        {
            Console.WriteLine("⚡ Training new model from house_data.csv...");

            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2f);

            var pipeline = mlContext.Transforms.Concatenate("Features", "Size", "Bedrooms", "Age")
                //.Append(mlContext.Transforms.NormalizeMinMax("Features")) // optional for small data
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 50));

            model = pipeline.Fit(split.TrainSet);
            mlContext.Model.Save(model, split.TrainSet.Schema, "HousePriceModel.zip");
            Console.WriteLine("✅ Model trained and saved to HousePriceModel.zip");
        }

        var predEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);

        // 3️⃣ Ollama setup
        var ollama = new OllamaApiClient(new Uri("http://localhost:11434"));
        ollama.SelectedModel = "qwen3:4b";

        Console.WriteLine("💬 House Price Predictor - type 'exit' to quit");

        Console.CancelKeyPress += (sender, e) =>
        {
            e.Cancel = true;
            Environment.Exit(0);
        };

        // 4️⃣ Chat loop
        while (true)
        {
            Console.Write("\nYou: ");
            var input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input) || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            // 4a️⃣ Ollama parses input to JSON directly
            var parseRequest = new ChatRequest
            {
                Messages = new List<Message>
                {
                    new Message
                    {
                        Role = OllamaChatRole.System,
                        Content = "You are an assistant that converts natural-language house descriptions into JSON with keys Size, Bedrooms, Age. Respond ONLY with JSON."
                    },
                    new Message
                    {
                        Role = OllamaChatRole.User,
                        Content = input
                    }
                }
            };

            string jsonOutput = "";
            await foreach (var chunk in ollama.ChatAsync(parseRequest))
            {
                if (chunk?.Message?.Content != null)
                    jsonOutput += chunk.Message.Content;
            }

            HouseData? house = null;
            try
            {
                house = JsonSerializer.Deserialize<HouseData>(jsonOutput);
            }
            catch
            {
                Console.WriteLine("[ERROR]: Could not parse house data. Please specify Size, Bedrooms, and Age.");
                continue;
            }

            if (house != null)
            {
                // 4b️⃣ Adjust to nearest bedroom in dataset
                var allData = mlContext.Data.CreateEnumerable<HouseData>(dataView, reuseRowObject: false).ToList();
                var nearestBedroomRow = allData.OrderBy(h => Math.Abs(h.Bedrooms - house.Bedrooms)).First();
                house.Bedrooms = nearestBedroomRow.Bedrooms;

                // 4c️⃣ ML.NET prediction
                var prediction = predEngine.Predict(house);

                // 4d️⃣ Ollama formats nicely
                var formatRequest = new ChatRequest
                {
                    Messages = new List<Message>
                    {
                        new Message
                        {
                            Role = OllamaChatRole.System,
                            Content = "You are an assistant that formats a numeric price into US dollars with commas. Respond ONLY with the formatted price, nothing else."
                        },
                        new Message
                        {
                            Role = OllamaChatRole.User,
                            Content = $"The predicted house price is {prediction.Price}."
                        }
                    }
                };

                string formattedPrice = "";
                await foreach (var chunk in ollama.ChatAsync(formatRequest))
                {
                    if (chunk?.Message?.Content != null)
                        formattedPrice += chunk.Message.Content;
                }

                Console.WriteLine($"[PREDICTOR]: {formattedPrice.Trim()}");
            }
        }

        Console.WriteLine("👋 Goodbye!");
    }
}
