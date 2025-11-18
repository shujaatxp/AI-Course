using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OllamaSharp;
using OllamaSharp.Models.Chat;
using OllamaChatRole = OllamaSharp.Models.Chat.ChatRole;

class Program
{
    static async Task Main()
    {
        var ollama = new OllamaApiClient(new Uri("http://localhost:11434"));
        ollama.SelectedModel = "qwen3:4b"; // Make sure this model exists locally

        Console.WriteLine("💬 ChatBot Ollama - type 'exit' to quit");

        // Handle Ctrl+C gracefully
        Console.CancelKeyPress += (sender, e) =>
        {
            e.Cancel = true;
            Environment.Exit(0);
        };

        while (true)
        {
            Console.Write("\nYou: ");
            var input = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(input) || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            var chatRequest = new ChatRequest
            {
                Messages = new List<Message>
                {
                    new Message { Role = OllamaChatRole.System, Content = "You are a helpful assistant." },
                    new Message { Role = OllamaChatRole.User, Content = input }
                }
            };

            try
            {
                Console.Write("[ASSISTANT]: ");
                var responseStream = ollama.ChatAsync(chatRequest);

                await foreach (var chunk in responseStream)
                {
                    if (chunk?.Message?.Content != null)
                    {
                        Console.Write(chunk.Message.Content); // Print chunks as they arrive
                    }
                }

                Console.WriteLine(); // End line after assistant finishes
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ Error: {ex.Message}");
            }
        }

        Console.WriteLine("👋 Goodbye!");
    }
}
