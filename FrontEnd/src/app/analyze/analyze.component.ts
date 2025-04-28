import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-analyze',
  templateUrl: './analyze.component.html',
  styleUrls: ['./analyze.component.scss']
})
export class AnalyzeComponent implements OnInit {

  /** 업로드된 파일 목록 */
  uploadedFiles: File[] = [];

  ngOnInit(): void {}

  /* ==============================
      Drag & Drop
  ============================== */
  onFileDrop(event: DragEvent): void {
    event.preventDefault();
    const files = event.dataTransfer?.files;
    if (files) { this.addFiles(files); }
  }

  onDragOver(event: DragEvent): void { event.preventDefault(); }
  onDragLeave(event: DragEvent): void { event.preventDefault(); }

  /* ==============================
      파일 선택창
  ============================== */
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files) { this.addFiles(input.files); }
  }

  /* ==============================
      공통 파일 추가 로직
  ============================== */
  private addFiles(fileList: FileList): void {
    Array.from(fileList).forEach(file => {
      // 중복 방지
      const exists = this.uploadedFiles
        .some(f => f.name === file.name && f.size === file.size);
      if (!exists) {
        this.uploadedFiles.push(file);
        console.log('업로드된 파일:', file);
      }
    });
  }
}
