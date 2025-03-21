"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Upload } from "lucide-react"

interface FileUploadProps {
  onTextExtracted: (text: string) => void
}

export default function FileUpload({ onTextExtracted }: FileUploadProps) {
  const [isUploading, setIsUploading] = useState(false)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)

    try {
      // TODO: Implement file upload and text extraction logic
      // This is a placeholder for the actual implementation
      await new Promise((resolve) => setTimeout(resolve, 1000))
      onTextExtracted(`Extracted text from ${file.name}`)
    } catch (error) {
      console.error("Error uploading file:", error)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="flex items-center space-x-2">
      <Input
        type="file"
        accept=".pdf,.png,.jpg,.jpeg"
        onChange={handleFileUpload}
        disabled={isUploading}
        className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
      />
      <Button type="button" variant="outline" disabled={isUploading}>
        <Upload className="h-4 w-4 mr-2" />
        {isUploading ? "Uploading..." : "Upload"}
      </Button>
    </div>
  )
}

