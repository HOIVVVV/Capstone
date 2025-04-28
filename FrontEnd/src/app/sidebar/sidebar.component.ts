import { Component, OnInit } from '@angular/core';

declare const $: any;

declare interface RouteInfo {
    path: string;
    title: string;
    icon: string;
    class: string;
}

// ✨ 새로운 하수관 관리용 메뉴 리스트
export const ROUTES: RouteInfo[] = [
  { path: '/dashboard', title: '대시보드', icon: 'pe-7s-graph', class: '' },
  { path: '/inspection', title: '관로 점검 결과', icon: 'pe-7s-search', class: '' },
  { path: '/damage-stats', title: '손상 통계', icon: 'pe-7s-graph3', class: '' },
  { path: '/mapping', title: '손상도 매핑', icon: 'pe-7s-map', class: '' },       // ✅ 추가
  { path: '/analyze', title: '하수관로 손상 분석', icon: 'pe-7s-graph2', class: '' } // ✅ 추가
];

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html'
})
export class SidebarComponent implements OnInit {
  menuItems: any[];

  constructor() { }

  ngOnInit() {
    this.menuItems = ROUTES.filter(menuItem => menuItem);
  }

  isMobileMenu() {
      if ($(window).width() > 991) {
          return false;
      }
      return true;
  };
}
