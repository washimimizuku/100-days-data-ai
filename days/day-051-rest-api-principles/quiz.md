# Day 51: REST API Principles - Quiz

Test your understanding of REST API design principles.

---

## Questions

### Question 1
What does REST stand for?

A) Remote Execution State Transfer  
B) Representational State Transfer  
C) Resource Execution Service Transfer  
D) Representational Service Technology

### Question 2
Which HTTP method is idempotent but NOT safe?

A) GET  
B) POST  
C) PUT  
D) OPTIONS

### Question 3
What HTTP status code should be returned when a resource is successfully created?

A) 200 OK  
B) 201 Created  
C) 204 No Content  
D) 202 Accepted

### Question 4
Which URL design follows REST best practices?

A) GET /api/getUsers  
B) GET /api/user/123  
C) GET /api/users/123  
D) GET /api/Users/123

### Question 5
What is the correct HTTP status code for "resource not found"?

A) 400 Bad Request  
B) 401 Unauthorized  
C) 403 Forbidden  
D) 404 Not Found

### Question 6
Which HTTP method should be used for partial updates?

A) PUT  
B) PATCH  
C) POST  
D) UPDATE

### Question 7
What is the recommended approach for API versioning?

A) Query parameter: /api/users?version=1  
B) Header: Accept: application/vnd.api.v1+json  
C) URL path: /api/v1/users  
D) Subdomain: v1.api.example.com

### Question 8
Which status code indicates rate limit exceeded?

A) 403 Forbidden  
B) 429 Too Many Requests  
C) 503 Service Unavailable  
D) 509 Bandwidth Limit Exceeded

### Question 9
What should a DELETE request return when successful with no content?

A) 200 OK with empty body  
B) 201 Created  
C) 204 No Content  
D) 202 Accepted

### Question 10
Which query parameter pattern is best for sorting in descending order?

A) ?sort=price&order=desc  
B) ?sort=-price  
C) ?sort=price:desc  
D) ?orderBy=price&direction=desc

---

## Answers

### Answer 1
**B) Representational State Transfer**

REST stands for Representational State Transfer, an architectural style for designing networked applications using HTTP protocol.

### Answer 2
**C) PUT**

PUT is idempotent (multiple identical requests have the same effect) but NOT safe (it modifies resource state). GET is both idempotent and safe. POST is neither idempotent nor safe.

### Answer 3
**B) 201 Created**

201 Created indicates a resource was successfully created. The response typically includes a Location header with the new resource's URL. 200 OK is for successful reads, 204 No Content for successful deletes.

### Answer 4
**C) GET /api/users/123**

REST best practices: use plural nouns (/users not /user), lowercase, no verbs in URL (users not getUsers), and consistent naming.

### Answer 5
**D) 404 Not Found**

404 Not Found indicates the requested resource doesn't exist. 400 is for bad request format, 401 for missing authentication, 403 for insufficient permissions.

### Answer 6
**B) PATCH**

PATCH is used for partial updates (updating specific fields). PUT replaces the entire resource. POST creates new resources. UPDATE is not a valid HTTP method.

### Answer 7
**C) URL path: /api/v1/users**

URL path versioning (/api/v1/users) is the most common and recommended approach. It's simple, visible, and easy to understand. Other methods work but are less common.

### Answer 8
**B) 429 Too Many Requests**

429 Too Many Requests specifically indicates rate limiting. Response should include Retry-After header. 403 is for authorization, 503 for service unavailable.

### Answer 9
**C) 204 No Content**

204 No Content indicates successful operation with no response body. Perfect for DELETE operations. 200 OK could be used but 204 is more semantically correct.

### Answer 10
**B) ?sort=-price**

Using a minus sign prefix (-price) for descending order is concise and widely adopted. It's used by many popular APIs. Multiple fields: ?sort=category,-price,name

---

## Scoring

- **10/10**: Perfect! You understand REST principles
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review HTTP methods and status codes
- **4-5/10**: Fair - Review REST constraints and best practices
- **0-3/10**: Needs work - Review all sections carefully

---

## Key Concepts to Remember

1. **REST Principles**: Stateless, client-server, cacheable, uniform interface
2. **HTTP Methods**: GET (read), POST (create), PUT (replace), PATCH (update), DELETE (remove)
3. **Status Codes**: 2xx success, 4xx client error, 5xx server error
4. **URL Design**: Use nouns, plural, lowercase, no verbs
5. **Idempotency**: PUT and DELETE are idempotent, POST is not
6. **Versioning**: URL path versioning is most common
7. **Pagination**: Include metadata (total, page, links)
8. **Error Handling**: Consistent format with error codes
