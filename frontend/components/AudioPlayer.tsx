"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Play, Pause, RotateCcw, Download } from "lucide-react"

interface AudioPlayerProps {
  audioUrl: string
}

export default function AudioPlayer({ audioUrl }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const audioRef = useRef<HTMLAudioElement>(null)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const setAudioData = () => {
      setDuration(audio.duration)
      setCurrentTime(audio.currentTime)
    }

    const setAudioTime = () => setCurrentTime(audio.currentTime)

    audio.addEventListener("loadeddata", setAudioData)
    audio.addEventListener("timeupdate", setAudioTime)

    return () => {
      audio.removeEventListener("loadeddata", setAudioData)
      audio.removeEventListener("timeupdate", setAudioTime)
    }
  }, [])

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleSliderChange = (value: number[]) => {
    if (audioRef.current) {
      audioRef.current.currentTime = value[0]
      setCurrentTime(value[0])
    }
  }

  const resetAudio = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0
      setCurrentTime(0)
      setIsPlaying(false)
    }
  }

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, "0")}`
  }

  return (
    <div className="mt-6 p-4 bg-indigo-50 rounded-lg shadow-inner animate-fade-in">
      <audio ref={audioRef} src={audioUrl} />
      <div className="flex items-center justify-between mb-2">
        <Button variant="outline" size="icon" onClick={togglePlay} className="hover:bg-indigo-100">
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <span className="text-sm font-medium text-indigo-800">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <Button variant="outline" size="icon" onClick={resetAudio} className="hover:bg-indigo-100">
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>
      <Slider value={[currentTime]} max={duration} step={0.1} onValueChange={handleSliderChange} className="w-full" />
      <Button variant="link" className="mt-2 text-indigo-600 hover:text-indigo-800 transition-colors">
        <Download className="h-4 w-4 mr-2" />
        Download Audio
      </Button>
    </div>
  )
}

