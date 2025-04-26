import { Component, OnInit } from '@angular/core';
import { ChartType } from '../lbd/lbd-chart/lbd-chart.component';

@Component({
  selector: 'app-damage-stats',
  templateUrl: './damage-stats.component.html',
  styleUrls: ['./damage-stats.component.css']
})
export class DamageStatsComponent implements OnInit {
  damageTypeChartData: any;
  damageTypeChartLegendItems: any[];

  sectionDamageChartData: any;
  sectionDamageChartLegendItems: any[];

  constructor() {}

  ngOnInit() {
    // 손상 유형 통계
    this.damageTypeChartData = {
      labels: ['균열', '침하', '누수', '이물질'],
      series: [25, 30, 20, 25]
    };
    this.damageTypeChartLegendItems = [
      { title: '균열', imageClass: 'fa fa-circle text-info' },
      { title: '침하', imageClass: 'fa fa-circle text-danger' },
      { title: '누수', imageClass: 'fa fa-circle text-warning' },
      { title: '이물질', imageClass: 'fa fa-circle text-success' }
    ];

    // 구간별 손상률
    this.sectionDamageChartData = {
      labels: ['구간 A', '구간 B', '구간 C', '구간 D'],
      series: [[10, 20, 40, 15]]
    };
    this.sectionDamageChartLegendItems = [
      { title: '손상률 (%)', imageClass: 'fa fa-circle text-primary' }
    ];
  }
}
