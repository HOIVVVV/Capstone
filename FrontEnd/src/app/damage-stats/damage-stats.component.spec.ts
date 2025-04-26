import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DamageStatsComponent } from './damage-stats.component';

describe('DamageStatsComponent', () => {
  let component: DamageStatsComponent;
  let fixture: ComponentFixture<DamageStatsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DamageStatsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DamageStatsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
