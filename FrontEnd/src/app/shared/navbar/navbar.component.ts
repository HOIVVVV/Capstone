import { Component, OnInit, ElementRef } from '@angular/core';
import { ROUTES } from '../../sidebar/sidebar.component';
import { Location } from '@angular/common';
import { Router } from '@angular/router';            // 🆕 라우터 불러오기

@Component({
  selector: 'navbar-cmp',
  templateUrl: 'navbar.component.html'
})
export class NavbarComponent implements OnInit {
  private listTitles: any[];
  location: Location;
  private toggleButton: any;
  private sidebarVisible = false;
  query = '';                                        // 🆕 검색어 바인딩 변수

  constructor(
    location: Location,
    private element: ElementRef,
    private router: Router                           // 🆕 라우터 주입
  ) {
    this.location = location;
  }

  ngOnInit(): void {
    this.listTitles = ROUTES.filter(listTitle => listTitle);
    const navbar: HTMLElement = this.element.nativeElement;
    this.toggleButton = navbar.getElementsByClassName('navbar-toggle')[0];
  }

  /* ───── 사이드바 토글 메서드 그대로 ───── */
  sidebarOpen(): void {
    setTimeout(() => this.toggleButton.classList.add('toggled'), 500);
    document.body.classList.add('nav-open');
    this.sidebarVisible = true;
  }

  sidebarClose(): void {
    this.toggleButton.classList.remove('toggled');
    document.body.classList.remove('nav-open');
    this.sidebarVisible = false;
  }

  sidebarToggle(): void {
    this.sidebarVisible ? this.sidebarClose() : this.sidebarOpen();
  }

  /* ───── 검색 실행 메서드 추가 ───── */
  onSearch(): void {                                 // 🆕
    const keyword = this.query.trim();
    if (!keyword) { return; }
    this.router.navigate(['/search-results'], { queryParams: { q: keyword } });
    this.query = '';                                 // 입력창 초기화
  }

  /* ───── 현재 페이지 타이틀 반환 ───── */
  getTitle(): string {
    let titlee = this.location.prepareExternalUrl(this.location.path());
    if (titlee.charAt(0) === '#') { titlee = titlee.slice(1); }

    for (let item = 0; item < this.listTitles.length; item++) {
      if (this.listTitles[item].path === titlee) {
        return this.listTitles[item].title;
      }
    }
    return 'Dashboard';
  }
}
