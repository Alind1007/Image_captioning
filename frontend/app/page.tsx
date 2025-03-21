import TextToAudioConverter from "../components/TextToAudioConverter"

export default function Home() {
  return (
    <main className="container mx-auto p-4 min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-indigo-100 to-purple-100">
      <h1 className="text-4xl font-bold mb-8 text-center text-indigo-800 animate-fade-in">Text to Audio Converter</h1>
      <TextToAudioConverter />
    </main>
  )
}

