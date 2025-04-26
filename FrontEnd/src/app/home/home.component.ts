import { Component, OnInit } from '@angular/core';
import { LocationStrategy, PlatformLocation, Location } from '@angular/common';
import { LegendItem, ChartType } from '../lbd/lbd-chart/lbd-chart.component';
import * as Chartist from 'chartist';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
    public emailChartType: ChartType;
    public emailChartData: any;
    public emailChartLegendItems: LegendItem[];

    public hoursChartType: ChartType;
    public hoursChartData: any;
    public hoursChartOptions: any;
    public hoursChartResponsive: any[];
    public hoursChartLegendItems: LegendItem[];

    public activityChartType: ChartType;
    public activityChartData: any;
    public activityChartOptions: any;
    public activityChartResponsive: any[];
    public activityChartLegendItems: LegendItem[];

    // ✨ 추가: 하수관로 손상 차트용 변수
    public damageTypeChartType: ChartType;
    public damageTypeChartData: any;
    public damageTypeChartLegendItems: LegendItem[];

    public sectionDamageChartType: ChartType;
    public sectionDamageChartData: any;
    public sectionDamageChartOptions: any;
    public sectionDamageChartResponsive: any[];
    public sectionDamageChartLegendItems: LegendItem[];

    public monthlyDamageChartType: ChartType;
    public monthlyDamageChartData: any;
    public monthlyDamageChartOptions: any;
    public monthlyDamageChartResponsive: any[];
    public monthlyDamageChartLegendItems: LegendItem[];

  constructor() { }

  ngOnInit() {
      // ✅ 기존 emailChart / hoursChart / activityChart 초기화
      this.emailChartType = ChartType.Pie;
      this.emailChartData = {
        labels: ['62%', '32%', '6%'],
        series: [62, 32, 6]
      };
      this.emailChartLegendItems = [
        { title: 'Open', imageClass: 'fa fa-circle text-info' },
        { title: 'Bounce', imageClass: 'fa fa-circle text-danger' },
        { title: 'Unsubscribe', imageClass: 'fa fa-circle text-warning' }
      ];

      this.hoursChartType = ChartType.Line;
      this.hoursChartData = {
        labels: ['9:00AM', '12:00AM', '3:00PM', '6:00PM', '9:00PM', '12:00PM', '3:00AM', '6:00AM'],
        series: [
          [287, 385, 490, 492, 554, 586, 698, 695, 752, 788, 846, 944],
          [67, 152, 143, 240, 287, 335, 435, 437, 539, 542, 544, 647],
          [23, 113, 67, 108, 190, 239, 307, 308, 439, 410, 410, 509]
        ]
      };
      this.hoursChartOptions = {
        low: 0,
        high: 800,
        showArea: true,
        height: '245px',
        axisX: {
          showGrid: false,
        },
        lineSmooth: Chartist.Interpolation.simple({
          divisor: 3
        }),
        showLine: false,
        showPoint: false,
      };
      this.hoursChartResponsive = [
        ['screen and (max-width: 640px)', {
          axisX: {
            labelInterpolationFnc: function (value) {
              return value[0];
            }
          }
        }]
      ];
      this.hoursChartLegendItems = [
        { title: 'Open', imageClass: 'fa fa-circle text-info' },
        { title: 'Click', imageClass: 'fa fa-circle text-danger' },
        { title: 'Click Second Time', imageClass: 'fa fa-circle text-warning' }
      ];

      this.activityChartType = ChartType.Bar;
      this.activityChartData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        series: [
          [542, 443, 320, 780, 553, 453, 326, 434, 568, 610, 756, 895],
          [412, 243, 280, 580, 453, 353, 300, 364, 368, 410, 636, 695]
        ]
      };
      this.activityChartOptions = {
        seriesBarDistance: 10,
        axisX: {
          showGrid: false
        },
        height: '245px'
      };
      this.activityChartResponsive = [
        ['screen and (max-width: 640px)', {
          seriesBarDistance: 5,
          axisX: {
            labelInterpolationFnc: function (value) {
              return value[0];
            }
          }
        }]
      ];
      this.activityChartLegendItems = [
        { title: 'Tesla Model S', imageClass: 'fa fa-circle text-info' },
        { title: 'BMW 5 Series', imageClass: 'fa fa-circle text-danger' }
      ];

      // ✨ 추가: 손상 통계용 차트 초기화
      this.damageTypeChartType = ChartType.Pie;
      this.damageTypeChartData = {
        labels: ['균열', '침하', '누수', '이물질'],
        series: [20, 30, 25, 25]
      };
      this.damageTypeChartLegendItems = [
        { title: '균열', imageClass: 'fa fa-circle text-info' },
        { title: '침하', imageClass: 'fa fa-circle text-danger' },
        { title: '누수', imageClass: 'fa fa-circle text-warning' },
        { title: '이물질', imageClass: 'fa fa-circle text-success' }
      ];

      this.sectionDamageChartType = ChartType.Line;
      this.sectionDamageChartData = {
        labels: ['구간 A', '구간 B', '구간 C', '구간 D'],
        series: [[5, 15, 40, 10]]
      };
      this.sectionDamageChartOptions = {
        low: 0,
        high: 50,
        showArea: true
      };
      this.sectionDamageChartResponsive = [
        ['screen and (max-width: 640px)', {
          seriesBarDistance: 5,
          axisX: {
            labelInterpolationFnc: function (value) {
              return value[0];
            }
          }
        }]
      ];
      this.sectionDamageChartLegendItems = [
        { title: '손상률 (%)', imageClass: 'fa fa-circle text-warning' }
      ];

      this.monthlyDamageChartType = ChartType.Bar;
      this.monthlyDamageChartData = {
        labels: ['1월', '2월', '3월', '4월', '5월'],
        series: [[3, 7, 8, 5, 2]]
      };
      this.monthlyDamageChartOptions = {
        axisX: {
          showGrid: false
        }
      };
      this.monthlyDamageChartResponsive = [
        ['screen and (max-width: 640px)', {
          seriesBarDistance: 5,
          axisX: {
            labelInterpolationFnc: function (value) {
              return value[0];
            }
          }
        }]
      ];
      this.monthlyDamageChartLegendItems = [
        { title: '손상 건수', imageClass: 'fa fa-circle text-danger' }
      ];
  }
}
