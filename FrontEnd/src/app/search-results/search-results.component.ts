import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-search-results',
  templateUrl: './search-results.component.html'
})
export class SearchResultsComponent {
  query = '';

  constructor(private route: ActivatedRoute) {
    this.route.queryParams.subscribe(p => this.query = p['q'] || '');
  }
}
