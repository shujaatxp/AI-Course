using System;
using System.Threading.Tasks;
using OpenAI;
using OpenAI.Chat;

class Program
{
    static async Task Main()
    {
        // Use your API key (hardcoded or from environment variable)
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY"); 
        var client = new OpenAIClient(apiKey);

        // Use a valid model. "gpt-5-nano" does NOT exist, switch to a valid one:
        var chatClient = client.GetChatClient("gpt-5-nano");

        Console.WriteLine("💬 ChatBot - type 'exit' to quit");

        while (true)
        {
            Console.Write("\nYou: ");
            var question = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(question) || question.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            // Send user question to GPT
            var response = await chatClient.CompleteChatAsync(
                ChatMessage.CreateUserMessage(question)
            );

            // Display GPT's answer
            var answer = response.Value.Content[0].Text;
            Console.WriteLine($"[ASSISTANT]: {answer}");
        }
    }
}
