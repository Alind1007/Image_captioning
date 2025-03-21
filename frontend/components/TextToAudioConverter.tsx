"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Card, CardContent } from "@/components/ui/card"
import AudioPlayer from "./AudioPlayer"
import FileUpload from "./FileUpload"
import { Loader2 } from "lucide-react"

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
    <Card className="w-full max-w-2xl shadow-lg animate-fade-in-up">
      <CardContent className="p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="text-input">Enter text or upload a file</Label>
            <Textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type your text here..."
              className="min-h-[100px] transition-all duration-200 focus:min-h-[150px]"
            />
          </div>
          <FileUpload onTextExtracted={setText} />
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="accent-select">Select accent</Label>
              <Select value={accent} onValueChange={setAccent}>
                <SelectTrigger id="accent-select">
                  <SelectValue placeholder="Choose accent" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="indian">English (Indian)</SelectItem>
                  <SelectItem value="australian">English (Australian)</SelectItem>
                  <SelectItem value="british">English (British)</SelectItem>
                  <SelectItem value="american">English (American)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="voice-select">Select voice</Label>
              <Select value={voice} onValueChange={setVoice}>
                <SelectTrigger id="voice-select">
                  <SelectValue placeholder="Choose voice" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="male">Male</SelectItem>
                  <SelectItem value="female">Female</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <Button type="submit" disabled={isLoading} className="w-full">
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Converting...
              </>
            ) : (
              "Generate Audio"
            )}
          </Button>
        </form>
        {error && <p className="text-red-500 mt-4">{error}</p>}
        {audioUrl && <AudioPlayer audioUrl={audioUrl} />}
      </CardContent>
    </Card>
  )
}

