"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import AudioPlayer from "./AudioPlayer"

export default function TextToAudioConverter() {
  const [text, setText] = useState("")
  const [accent, setAccent] = useState("")
  const [voice, setVoice] = useState("")
  const [audioUrl, setAudioUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    try {
      // TODO: Replace this with actual API call to your model
      // const response = await fetch('/api/convert', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ text, accent, voice }),
      // })
      // if (!response.ok) throw new Error('Conversion failed')
      // const data = await response.json()
      // setAudioUrl(data.audioUrl)

      // Simulating API call
      await new Promise((resolve) => setTimeout(resolve, 2000))
      setAudioUrl("/placeholder.mp3")
    } catch (err) {
      setError("An error occurred during conversion. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to convert"
          required
          className="w-full"
        />
        <div className="flex space-x-4">
          <Select value={accent} onValueChange={setAccent} required>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select accent" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="indian">English (Indian)</SelectItem>
              <SelectItem value="australian">English (Australian)</SelectItem>
              <SelectItem value="british">English (British)</SelectItem>
              <SelectItem value="american">English (American)</SelectItem>
            </SelectContent>
          </Select>
          <Select value={voice} onValueChange={setVoice} required>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select voice" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="male">Male</SelectItem>
              <SelectItem value="female">Female</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button type="submit" disabled={isLoading} className="w-full">
          {isLoading ? "Converting..." : "Convert to Audio"}
        </Button>
      </form>
      {error && <p className="text-red-500 mt-4">{error}</p>}
      {audioUrl && <AudioPlayer audioUrl={audioUrl} />}
    </div>
  )
}

